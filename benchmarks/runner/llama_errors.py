"""llama-server b8932's error envelope, read into fixed labels (session 115, parts 19 and 20). Operator tooling, shared
by the reproduction, the verification and the step-4 workload, so that a failed request keeps a diagnosable category
without its message.

The envelope is ``{"error": {"code": N, "message": "...", "type": "..."}}`` (tools/server/server-common.cpp,
format_error_response). What is kept of it: the type when it is one of the build's fixed strings; a category matched
against the build's known messages; whether the message carries the grammar prefill's text, and whether the piece it
names begins the chat template's generation prompt; and the message's SHA-256 and length. The message itself is never
returned. A message equal to ``Failed to initialize samplers: std::exception`` is the sampler's prefill failure as the
live server reports it (common/sampling.cpp rethrows the caught exception by value, which slices its text; session 115,
part 19's reproduction matched that string by hash), so it is flagged ``sampler_init_sliced``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

# tools/server/server-common.cpp, format_error_response: its fixed type strings
ERROR_TYPES = frozenset({"invalid_request_error", "authentication_error", "not_found_error", "server_error",
                         "permission_error", "not_supported_error", "unavailable_error",
                         "exceed_context_size_error"})
# the build's known error messages, by the code that raises them; a message is matched, never kept
ERROR_CATEGORIES = (
    ("sampler_init_failed", "Failed to initialize samplers: "),  # server-context.cpp, launch_slot_with_task
    ("json_schema_conversion_failed", '"json_schema": '),  # server-task.cpp, params_from_json_cmpl
    ("json_schema_and_grammar", "Cannot use both json_schema and grammar"),  # server-common.cpp
    ("exceeds_context_size", "exceeds the available context size"),  # server-context.cpp
    ("image_processing_failed", "failed to process image"),  # server-context.cpp
)
CATEGORY_LABELS = (*(name for name, _ in ERROR_CATEGORIES), "other", "unparsable", "unreadable")
# llama-grammar.cpp (llama_grammar_accept_token), reached from common_sampler_init's prefill of the generation prompt
GRAMMAR_PREFILL = "Unexpected empty grammar stack after accepting piece: "
GENERATION_PROMPT_START = "<|im_start|>"  # both LFM2-VL templates' generation prompt (scene_smoke's chatml_problem)
SAMPLER_INIT_SLICED = "Failed to initialize samplers: std::exception"


def error_facts(payload: Any) -> dict[str, Any]:
    """The error envelope's fixed type and the message's category, without the message."""
    envelope = payload.get("error") if isinstance(payload, Mapping) else None
    envelope = envelope if isinstance(envelope, Mapping) else {}
    message = envelope.get("message")
    message = message if isinstance(message, str) else ""
    kind = envelope.get("type")
    facts: dict[str, Any] = {
        "type": kind if kind in ERROR_TYPES else "other", "envelope": bool(envelope),
        "category": next((name for name, text in ERROR_CATEGORIES if text in message), "other"),
        "grammar_prefill_rejected": GRAMMAR_PREFILL in message,
        "sampler_init_sliced": message == SAMPLER_INIT_SLICED,
        "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(), "message_chars": len(message),
    }
    if facts["grammar_prefill_rejected"]:
        piece = message[message.index(GRAMMAR_PREFILL) + len(GRAMMAR_PREFILL):]
        facts["piece_is_generation_prompt_start"] = piece.startswith(GENERATION_PROMPT_START)
    return facts


def error_category(data: bytes) -> str:
    """One fixed label for an HTTP error's body: its category, or ``unparsable`` when the body is not JSON."""
    try:
        payload = json.loads(data)
    except ValueError:
        return "unparsable"
    return error_facts(payload)["category"]
