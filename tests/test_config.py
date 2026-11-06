import os
import unittest
from sentinel.core.config import (
    BiometricConfig,
    DetectionConfig,
    IntelligenceConfig,
    SentinelConfig,
    StreamConfig,
)


class TestSentinelConfig(unittest.TestCase):

    def test_detection_config_threshold_validation(self):
        """conf_threshold must be clamped between 0.0 and 1.0."""
        with self.assertRaises(ValueError):
            DetectionConfig(conf_threshold=1.5).validate()
        with self.assertRaises(ValueError):
            DetectionConfig(conf_threshold=-0.1).validate()

    def test_biometric_config_threshold_validation(self):
        """match_threshold and reid_threshold must be between 0.0 and 1.0."""
        with self.assertRaises(ValueError):
            BiometricConfig(match_threshold=2.0).validate()
        with self.assertRaises(ValueError):
            BiometricConfig(reid_threshold=-0.5).validate()

    def test_intelligence_config_restricted_hours_validation(self):
        """Restricted hours must be valid clock hours (0-23)."""
        with self.assertRaises(ValueError):
            IntelligenceConfig(restricted_hours=(25, 6)).validate()
        with self.assertRaises(ValueError):
            IntelligenceConfig(restricted_hours=(22, 24)).validate()

    def test_fail_fast_on_missing_rtsp(self):
        """SentinelConfig.load_and_validate must raise ValueError if RTSP_URL is empty."""
        os.environ["RTSP_URL"] = ""
        with self.assertRaises(ValueError) as ctx:
            SentinelConfig.load_and_validate(enforce_secrets=True)
        self.assertIn("RTSP_URL is required", str(ctx.exception))

    def test_load_testing_mode_without_secrets(self):
        """SentinelConfig can load safely in testing mode (enforce_secrets=False)."""
        os.environ["RTSP_URL"] = ""
        cfg = SentinelConfig.load_and_validate(enforce_secrets=False)
        self.assertEqual(cfg.detection.conf_threshold, 0.4)
        self.assertTrue(cfg.headless)
        self.assertEqual(cfg.intelligence.restricted_hours, (22, 6))


if __name__ == "__main__":
    unittest.main()
