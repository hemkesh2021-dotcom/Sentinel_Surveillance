"""JPEG pictures for the live preview and incident snapshots (D69, D70).

``encode_jpeg`` is the device encoder. It imports cv2 (the runtime's OpenCV in
``~/onvif_env``) only when called, so portable code and tests never load it;
tests pass their own encoder. Every bound is explicit: a picture is scaled
down, never up, to fit ``max_width`` x ``max_height``, on a copy (the frame
itself is never written), at a fixed quality, and callers check the byte
size. Problems are fixed labels, never exception text.

``with_comment`` adds a JPEG comment (COM) segment without touching the
pixels, so a downloaded snapshot names its incident, frame and time.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..contracts import PixelFormat

SOI = b"\xff\xd8"
APP0 = b"\xff\xe0"
COM = b"\xff\xfe"
SOS = b"\xff\xda"
COMMENT_MAX_CHARS = 512
_PRINTABLE = re.compile(r"[ -~]{1,%d}" % COMMENT_MAX_CHARS)


class EncodeError(Exception):
    """A picture could not be encoded; ``label`` is a fixed snake_case reason."""

    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


@dataclass(frozen=True)
class EncodedJpeg:
    data: bytes
    width: int
    height: int


Encoder = Callable[..., EncodedJpeg]  # (image, pixel_format, *, max_width, max_height, quality) -> EncodedJpeg


def fit_within(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    """The size ``width`` x ``height`` is scaled to (down only, aspect kept) to fit ``max_width`` x ``max_height``."""
    if min(width, height, max_width, max_height) < 1:
        raise ValueError("sizes must be positive")
    if width <= max_width and height <= max_height:
        return width, height
    scale = min(max_width / width, max_height / height)
    return max(1, min(max_width, int(width * scale))), max(1, min(max_height, int(height * scale)))


def encode_jpeg(image: Any, pixel_format: PixelFormat, *, max_width: int, max_height: int, quality: int) -> EncodedJpeg:
    """Device encoder: ``image`` (a BGR, RGB or GRAY8 array) as a JPEG of at most ``max_width`` x ``max_height``."""
    import cv2  # device only (~/onvif_env); never imported by portable code paths

    shape = getattr(image, "shape", None)
    if shape is None or len(shape) not in (2, 3) or min(int(shape[0]), int(shape[1])) < 1:
        raise EncodeError("not_an_image")
    height, width = int(shape[0]), int(shape[1])
    channels = 1 if len(shape) == 2 else int(shape[2])
    if pixel_format is PixelFormat.BGR or pixel_format is PixelFormat.RGB:
        if channels != 3:
            raise EncodeError("unexpected_channels")
        if pixel_format is PixelFormat.RGB:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # a copy
    elif pixel_format is PixelFormat.GRAY8:
        if channels != 1:
            raise EncodeError("unexpected_channels")
    else:
        raise EncodeError("unsupported_pixel_format")
    out_width, out_height = fit_within(width, height, max_width, max_height)
    if (out_width, out_height) != (width, height):
        image = cv2.resize(image, (out_width, out_height), interpolation=cv2.INTER_AREA)  # a copy
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise EncodeError("jpeg_encode_failed")
    return EncodedJpeg(buffer.tobytes(), out_width, out_height)


def with_comment(jpeg: bytes, text: str) -> bytes:
    """``jpeg`` with ``text`` (printable ASCII) in a COM segment after SOI and any APP0 (JFIF) header."""
    if not _PRINTABLE.fullmatch(text):
        raise ValueError(f"a comment is 1-{COMMENT_MAX_CHARS} printable ASCII characters")
    if jpeg[:2] != SOI:
        raise EncodeError("not_a_jpeg")
    at = 2
    if jpeg[2:4] == APP0:
        if len(jpeg) < 6:
            raise EncodeError("not_a_jpeg")
        at = 4 + int.from_bytes(jpeg[4:6], "big")
        if at > len(jpeg):
            raise EncodeError("not_a_jpeg")
    payload = text.encode("ascii")
    return jpeg[:at] + COM + (len(payload) + 2).to_bytes(2, "big") + payload + jpeg[at:]


def jpeg_comments(jpeg: bytes, *, limit: int = 8) -> list[str]:
    """The COM segments before the image data (at most ``limit``); [] for anything that is not a JPEG."""
    if jpeg[:2] != SOI:
        return []
    comments: list[str] = []
    at = 2
    while at + 4 <= len(jpeg) and len(comments) < limit:
        marker = jpeg[at:at + 2]
        if marker[0] != 0xFF or marker == SOS:
            break
        length = int.from_bytes(jpeg[at + 2:at + 4], "big")
        if length < 2 or at + 2 + length > len(jpeg):
            break
        if marker == COM:
            comments.append(jpeg[at + 4:at + 2 + length].decode("ascii", "replace"))
        at += 2 + length
    return comments
