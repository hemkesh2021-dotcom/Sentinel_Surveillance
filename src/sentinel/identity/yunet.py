"""The pinned YuNet detector's score threshold: its one definition (review A, finding 10).

DeepFace 0.0.99's default ``yunet_score_threshold``. Production detection keeps the faces scoring at least this
(legacy_deepface), and the validation diagnostics count, compare and measure the full-scale candidates at the same
score so that they compare like for like with production: the D86 counts (``runtime.FULL_SCALE_COUNTED``), the D87
comparison's eligibility (``comparison.FULL_SCALE_COUNTED``) and the landmark records (``landmarks.LANDMARK_SCORE``).
Each of those names is this value, imported, never a copy; changing it here changes all four.

It is not the 0.6 quality gate (``identity.min_quality``), the identity match threshold (``identity.match_threshold``)
or the full-scale pass's recording floor (``legacy_deepface.FULL_SCALE_FLOOR``), which are separate settings.

This module imports nothing, so any module can import it without weight or an import cycle.
"""

YUNET_THRESHOLD = 0.9
