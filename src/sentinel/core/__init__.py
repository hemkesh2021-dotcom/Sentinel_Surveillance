"""Core domain types and configuration for Sentinel."""

from .config import SentinelConfig
from .types import (
    BoundingBox,
    FaceIdentity,
    SystemState,
    SystemStatus,
    ThreatAssessment,
    ThreatLevel,
    TrackedPerson,
)

__all__ = [
    "SentinelConfig",
    "BoundingBox",
    "FaceIdentity",
    "SystemState",
    "SystemStatus",
    "ThreatAssessment",
    "ThreatLevel",
    "TrackedPerson",
]
