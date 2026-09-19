import unittest
import numpy as np
from sentinel.core.types import (
    BoundingBox,
    SystemState,
    SystemStatus,
    ThreatLevel,
)


class TestSentinelTypes(unittest.TestCase):

    def test_bounding_box_geometry(self):
        """Verify width, height, and center coordinate calculations."""
        box = BoundingBox(x1=10, y1=20, x2=110, y2=220)
        self.assertEqual(box.width, 100)
        self.assertEqual(box.height, 200)
        self.assertEqual(box.center, (60, 120))

    def test_bounding_box_from_xyxy(self):
        """Factory method should accept numpy arrays and floats."""
        raw = np.array([5.2, 10.8, 55.1, 110.9])
        box = BoundingBox.from_xyxy(raw)
        self.assertEqual(box, BoundingBox(5, 10, 55, 110))

    def test_bounding_box_tuple_unpacking(self):
        """BoundingBox must unpack like a tuple (x1, y1, x2, y2) for backward compatibility."""
        box = BoundingBox(10, 20, 30, 40)
        x1, y1, x2, y2 = box
        self.assertEqual((x1, y1, x2, y2), (10, 20, 30, 40))

    def test_bounding_box_crop_from(self):
        """Safe crop should pad and clamp within image boundaries without crashing."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        box = BoundingBox(x1=90, y1=90, x2=120, y2=120)  # Partially outside
        crop = box.crop_from(frame, padding=5)
        self.assertGreater(crop.shape[0], 0)
        self.assertGreater(crop.shape[1], 0)

    def test_bitmask_compound_status(self):
        """IntFlag allows multiple statuses (FIRE | INTRUDER)."""
        status = SystemStatus.INTRUDER | SystemStatus.FIRE
        self.assertTrue(bool(status & SystemStatus.FIRE))
        self.assertTrue(bool(status & SystemStatus.INTRUDER))
        self.assertEqual(status.to_display_str(), "FIRE & INTRUDER!")

    def test_threat_level_from_str(self):
        """Case-insensitive parser for ThreatLevel."""
        self.assertEqual(ThreatLevel.from_str("HIGH"), ThreatLevel.HIGH)
        self.assertEqual(ThreatLevel.from_str("  medium  "), ThreatLevel.MEDIUM)
        self.assertEqual(ThreatLevel.from_str("unknown_val"), ThreatLevel.NONE)

    def test_system_state_serialization(self):
        """State object cleanly serializes to dict for IPC or dashboard."""
        state = SystemState(
            status=SystemStatus.INTRUDER | SystemStatus.FIRE,
            fps=29.97,
        )
        data = state.to_dict()
        self.assertEqual(data["status"], "FIRE & INTRUDER!")
        self.assertEqual(data["status_flags"], 24)  # 8 | 16
        self.assertEqual(data["fps"], 30.0)


if __name__ == "__main__":
    unittest.main()
