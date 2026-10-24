"""
src/sentinel/core/types.py
Immutable domain models, optimized data schemas, and bitmask status flags for SENTINEL.
"""
from dataclasses import dataclass, field
from enum import Enum, IntFlag
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Represents pixel coordinates of a detection bounding box (x1, y1, x2, y2)."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def __iter__(self):
        """Allows direct unpacking: x1, y1, x2, y2 = box"""
        return iter((self.x1, self.y1, self.x2, self.y2))

    def to_dict(self) -> Dict[str, int]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @classmethod
    def from_xyxy(cls, coords: Any) -> "BoundingBox":
        """Constructs BoundingBox from list, tuple, or numpy array [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = coords
        return cls(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))

    def crop_from(self, frame: np.ndarray, padding: int = 0) -> np.ndarray:
        """Safely extracts the bounding box region from an image with optional padding."""
        h, w = frame.shape[:2]
        x1 = max(0, self.x1 - padding)
        y1 = max(0, self.y1 - padding)
        x2 = min(w, self.x2 + padding)
        y2 = min(h, self.y2 + padding)
        if x2 <= x1 or y2 <= y1:
            return frame[0:0, 0:0]
        return frame[y1:y2, x1:x2]


class ThreatLevel(str, Enum):
    """Normalized threat classification levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def from_str(cls, value: str) -> "ThreatLevel":
        val = str(value or "none").lower().strip()
        for member in cls:
            if member.value == val:
                return member
        return cls.NONE


class SystemStatus(IntFlag):
    """Bitmask system operational status allowing compound emergency states."""
    STARTING   = 0
    MONITORING = 1 << 0   # 1
    KNOWN      = 1 << 1   # 2
    STRANGER   = 1 << 2   # 4
    INTRUDER   = 1 << 3   # 8
    FIRE       = 1 << 4   # 16
    ERROR      = 1 << 5   # 32

    def to_display_str(self) -> str:
        """Converts bitmask flags to clean UI text."""
        if self & SystemStatus.FIRE and self & SystemStatus.INTRUDER:
            return "FIRE & INTRUDER!"
        if self & SystemStatus.FIRE:
            return "FIRE!"
        if self & SystemStatus.INTRUDER:
            return "INTRUDER"
        if self & SystemStatus.STRANGER:
            return "STRANGER"
        if self & SystemStatus.KNOWN:
            return "KNOWN"
        if self & SystemStatus.MONITORING:
            return "MONITORING"
        if self & SystemStatus.ERROR:
            return "ERROR"
        return "STARTING"


@dataclass(slots=True)
class FaceIdentity:
    """Biometric identity details associated with a track."""
    name: str = "Unknown"
    confidence: float = 0.0
    is_verified: bool = False
    is_stranger: bool = True
    retry_count: int = 0
    embedding: Optional[np.ndarray] = None


@dataclass(slots=True)
class TrackedPerson:
    """State of an actively tracked individual across frames."""
    track_id: int
    bbox: BoundingBox
    identity: FaceIdentity = field(default_factory=FaceIdentity)
    confirm_count: int = 0
    last_seen_frame: int = 0
    is_visible: bool = False


@dataclass(slots=True)
class ThreatAssessment:
    """Structured output from the Vision-Language Model (LFM2-VL)."""
    threat: ThreatLevel = ThreatLevel.NONE
    description: str = ""
    activities: List[str] = field(default_factory=list)
    postures: List[str] = field(default_factory=lambda: ["normal"])
    harmful: bool = False
    fire_smoke: bool = False
    persons_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threat": self.threat.value,
            "description": self.description,
            "activities": self.activities,
            "postures": self.postures,
            "harmful": self.harmful,
            "fire_smoke": self.fire_smoke,
            "persons": self.persons_count,
        }


@dataclass(slots=True)
class SystemState:
    """Atomic snapshot of the entire surveillance system state."""
    status: SystemStatus = SystemStatus.STARTING
    threat: ThreatAssessment = field(default_factory=ThreatAssessment)
    persons: List[Dict[str, Any]] = field(default_factory=list)
    fire: bool = False
    fps: float = 0.0
    room_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.to_display_str(),
            "status_flags": int(self.status),
            "threat": self.threat.threat.value,
            "description": self.threat.description,
            "persons": self.persons,
            "fire": self.fire,
            "fps": round(self.fps, 1),
            "room_count": self.room_count,
        }


if __name__ == "__main__":
    print("[TEST] Running self-test for Optimized Domain Models...")

    # 1. Test BoundingBox math with slots
    raw_coords = np.array([10, 20, 110, 220])
    box = BoundingBox.from_xyxy(raw_coords)
    assert box.width == 100
    assert box.height == 200
    assert box.center == (60, 120)
    print(f"[PASS] BoundingBox slots test passed: {box}")

    # 2. Test Compound Bitmask Status
    compound_status = SystemStatus.INTRUDER | SystemStatus.FIRE
    assert bool(compound_status & SystemStatus.FIRE) is True
    assert bool(compound_status & SystemStatus.INTRUDER) is True
    assert compound_status.to_display_str() == "FIRE & INTRUDER!"
    print(f"[PASS] Bitmask Compound Status test passed: {compound_status.to_display_str()}")

    # 3. Test SystemState Serialization
    state = SystemState(status=compound_status, fps=30.0)
    d = state.to_dict()
    assert d["status"] == "FIRE & INTRUDER!"
    assert d["status_flags"] == 24  # 8 | 16 = 24
    print(f"[PASS] SystemState serialization test passed: {d}")
