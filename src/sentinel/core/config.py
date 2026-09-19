"""
src/sentinel/core/config.py
Type-safe, fail-fast configuration engine for SENTINEL.
"""
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class StreamConfig:
    """RTSP video ingestion and camera stream properties."""
    rtsp_url: str = field(default_factory=lambda: os.getenv("RTSP_URL", ""))
    display_width: int = 640
    display_height: int = 480
    reconnect_delay_sec: float = 2.0

    def validate(self) -> None:
        if not self.rtsp_url:
            raise ValueError("[CONFIG ERROR] RTSP_URL is required and cannot be empty.")
        if self.display_width <= 0 or self.display_height <= 0:
            raise ValueError("[CONFIG ERROR] Frame dimensions must be positive integers.")
        if self.reconnect_delay_sec <= 0:
            raise ValueError("[CONFIG ERROR] reconnect_delay_sec must be positive.")


@dataclass(frozen=True)
class DetectionConfig:
    """YOLOv8 inference and multi-object tracking settings."""
    model_path: str = field(
        default_factory=lambda: os.getenv("YOLO_MODEL", str(Path.home() / "yolov8n.engine"))
    )
    conf_threshold: float = 0.40
    confirm_frames: int = 2
    persist_frames: int = 8
    frame_skip: int = 2
    tracker_config: str = "bytetrack.yaml"

    def validate(self) -> None:
        if not (0.0 <= self.conf_threshold <= 1.0):
            raise ValueError(
                f"[CONFIG ERROR] conf_threshold must be between 0.0 and 1.0, got {self.conf_threshold}"
            )
        if self.confirm_frames < 1:
            raise ValueError("[CONFIG ERROR] confirm_frames must be >= 1.")
        if self.persist_frames < 1:
            raise ValueError("[CONFIG ERROR] persist_frames must be >= 1.")
        if self.frame_skip < 1:
            raise ValueError("[CONFIG ERROR] frame_skip must be >= 1.")


@dataclass(frozen=True)
class BiometricConfig:
    """Biometric face detection, embedding generation, and Re-ID cache."""
    face_db_path: str = field(
        default_factory=lambda: os.getenv("FACE_DB_PATH", str(Path.home() / "face_db.pkl"))
    )
    model_name: str = "Facenet512"
    detector_backend: str = "yunet"
    match_threshold: float = 0.40
    reid_threshold: float = 0.48
    max_session_embeddings: int = 5
    stranger_recheck_interval_sec: float = 5.0
    max_retry_attempts: int = 15

    def validate(self) -> None:
        if not (0.0 <= self.match_threshold <= 1.0):
            raise ValueError(
                f"[CONFIG ERROR] match_threshold must be between 0.0 and 1.0, got {self.match_threshold}"
            )
        if not (0.0 <= self.reid_threshold <= 1.0):
            raise ValueError(
                f"[CONFIG ERROR] reid_threshold must be between 0.0 and 1.0, got {self.reid_threshold}"
            )
        if self.max_session_embeddings < 1:
            raise ValueError("[CONFIG ERROR] max_session_embeddings must be >= 1.")
        if self.max_retry_attempts < 1:
            raise ValueError("[CONFIG ERROR] max_retry_attempts must be >= 1.")


@dataclass(frozen=True)
class IntelligenceConfig:
    """Vision-Language Model (VLM) scene analysis & Telegram alerting parameters."""
    lfm2_server_url: str = field(
        default_factory=lambda: os.getenv("LFM2_SERVER", "http://localhost:8080")
    )
    lfm2_interval_sec: float = 4.0
    lfm2_img_width: int = 480
    lfm2_img_height: int = 360
    lfm2_img_quality: int = 60
    restricted_hours: Tuple[int, int] = (22, 6)
    intruder_log_path: str = field(
        default_factory=lambda: os.getenv("INTRUDER_LOG", str(Path.home() / "intruder_log.json"))
    )
    telegram_bot_token: str = field(
        default_factory=lambda: os.getenv("BOT_TOKEN", os.getenv("TELEGRAM_BOT_TOKEN", ""))
    )
    telegram_chat_id: str = field(
        default_factory=lambda: os.getenv("CHAT_ID", os.getenv("TELEGRAM_CHAT_ID", ""))
    )

    def validate(self) -> None:
        start_h, end_h = self.restricted_hours
        if not (0 <= start_h <= 23 and 0 <= end_h <= 23):
            raise ValueError(
                f"[CONFIG ERROR] restricted_hours must be (0-23, 0-23), got {self.restricted_hours}"
            )
        if self.lfm2_interval_sec <= 0:
            raise ValueError("[CONFIG ERROR] lfm2_interval_sec must be positive.")


@dataclass(frozen=True)
class IPCConfig:
    """Inter-process communication and state persistence file paths."""
    state_file: str = field(
        default_factory=lambda: os.getenv("STATE_FILE", "/tmp/surv_state.json")
    )
    frame_file: str = field(
        default_factory=lambda: os.getenv("FRAME_FILE", "/tmp/surv_frame.jpg")
    )
    state_write_interval_sec: float = 1.0


@dataclass(frozen=True)
class SentinelConfig:
    """Master immutable configuration container."""
    stream: StreamConfig = field(default_factory=StreamConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    biometric: BiometricConfig = field(default_factory=BiometricConfig)
    intelligence: IntelligenceConfig = field(default_factory=IntelligenceConfig)
    ipc: IPCConfig = field(default_factory=IPCConfig)
    headless: bool = field(
        default_factory=lambda: os.getenv("SENTINEL_HEADLESS", "1").lower() in ("1", "true", "yes")
    )

    @classmethod
    def load_and_validate(cls, enforce_secrets: bool = True) -> "SentinelConfig":
        """
        Factory method: Builds configuration from environment and runs fail-fast checks.

        Args:
            enforce_secrets: If True, requires RTSP_URL and Telegram tokens.
        """
        cfg = cls()
        if enforce_secrets:
            cfg.stream.validate()
            if not cfg.intelligence.telegram_bot_token or not cfg.intelligence.telegram_chat_id:
                raise ValueError("[CONFIG ERROR] Telegram BOT_TOKEN and CHAT_ID are required.")
        cfg.detection.validate()
        cfg.biometric.validate()
        cfg.intelligence.validate()
        return cfg


if __name__ == "__main__":
    print("Running self-test for SentinelConfig...")

    # Test 1: Fail-fast on empty RTSP
    os.environ["RTSP_URL"] = ""
    try:
        SentinelConfig.load_and_validate(enforce_secrets=True)
        print("FAILED: Should have raised ValueError for missing RTSP_URL")
    except ValueError as e:
        print(f"PASSED (Fail-fast validation caught missing secret): {e}")

    # Test 2: Successful instantiation without secrets enforced (testing mode)
    test_cfg = SentinelConfig.load_and_validate(enforce_secrets=False)
    print(f"PASSED (Config Loaded): Headless={test_cfg.headless}, YOLO Conf={test_cfg.detection.conf_threshold}")
