"""Telegram notifier (guide chapter 10; V2-15, demo form). Standard library only.

Success requires HTTP 200 **and** a JSON body with ``ok: true``; anything else
is a failure with a reason. Rate limits (HTTP 429 or ``error_code`` 429) are
retried no earlier than Telegram's ``parameters.retry_after``. Credential and
destination errors (400, 401, 403, 404) are permanent until an operator fixes
the configuration. Server errors and network failures are retried. A timeout or
a reset after the request was sent, or a 200 whose body cannot be read, is
*ambiguous*: Telegram may have delivered the message, and a retry may
duplicate it. [Telegram response contract](https://core.telegram.org/bots/api#making-requests)

The bot token is part of the request URL, so no URL, request or raw exception
text is ever returned: details are built from status codes, Telegram's
``description`` (redacted) and exception class names. The token and chat ID
come from the environment of the runtime process, never from the config file.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from collections.abc import Callable

from ..redaction import redact_line
from .outbox import DeliveryResult, DeliveryStatus, OutboxMessage

API_BASE = "https://api.telegram.org"
TOKEN_ENV = "SENTINEL_TELEGRAM_BOT_TOKEN"
CHAT_ENV = "SENTINEL_TELEGRAM_CHAT_ID"
MAX_RESPONSE_BYTES = 65536
PERMANENT_CODES = frozenset({400, 401, 403, 404})

Opener = Callable[..., object]


class NotifierUnavailable(Exception):
    """The channel cannot be used (missing credentials); its rows wait, nothing is sent."""


class TelegramNotifier:
    channel = "telegram"

    def __init__(
        self,
        token: str,
        chat_id: str,
        *,
        timeout_s: float = 10.0,
        opener: Opener = urllib.request.urlopen,
        api_base: str = API_BASE,
    ) -> None:
        if not token or not chat_id:
            raise NotifierUnavailable("telegram needs a bot token and a chat ID")
        self._token = token
        self._chat_id = chat_id
        self._timeout_s = timeout_s
        self._opener = opener
        self._api_base = api_base.rstrip("/")

    @classmethod
    def from_environment(cls, *, timeout_s: float = 10.0, environ: dict[str, str] | None = None) -> TelegramNotifier:
        env = os.environ if environ is None else environ
        token, chat = env.get(TOKEN_ENV, ""), env.get(CHAT_ENV, "")
        if not token or not chat:
            raise NotifierUnavailable(f"set {TOKEN_ENV} and {CHAT_ENV} in the runtime's environment")
        return cls(token, chat, timeout_s=timeout_s)

    def __repr__(self) -> str:  # never show the token or chat ID
        return "TelegramNotifier(<credentials hidden>)"

    def send(self, message: OutboxMessage) -> DeliveryResult:
        body = json.dumps(
            {"chat_id": self._chat_id, "text": message.text, "disable_web_page_preview": True}
        ).encode()
        request = urllib.request.Request(
            f"{self._api_base}/bot{self._token}/sendMessage",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self._timeout_s) as response:
                status = response.status
                raw = response.read(MAX_RESPONSE_BYTES)
        except urllib.error.HTTPError as exc:
            status = exc.code
            try:
                raw = exc.read(MAX_RESPONSE_BYTES)
            except Exception:  # noqa: BLE001 - the body is optional
                raw = b""
        except (TimeoutError, socket.timeout):
            return self._ambiguous("timeout")
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, (TimeoutError, socket.timeout)):
                return self._ambiguous("timeout")
            if isinstance(reason, ConnectionResetError):
                return self._ambiguous("connection reset")
            # DNS failure, refused connection, TLS failure: the request never reached Telegram.
            return DeliveryResult(DeliveryStatus.RETRY, f"network error: {type(reason).__name__}")
        except ConnectionResetError:
            return self._ambiguous("connection reset")
        except OSError as exc:
            return DeliveryResult(DeliveryStatus.RETRY, f"network error: {type(exc).__name__}", ambiguous=True)
        return self._interpret(status, raw)

    def _ambiguous(self, what: str) -> DeliveryResult:
        return DeliveryResult(
            DeliveryStatus.RETRY,
            f"{what} after sending; Telegram may have delivered the message",
            ambiguous=True,
        )

    @staticmethod
    def _interpret(status: int, raw: bytes) -> DeliveryResult:
        try:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("not an object")
        except (UnicodeDecodeError, ValueError):
            if status == 200:
                return DeliveryResult(
                    DeliveryStatus.RETRY, "HTTP 200 with an unreadable body; delivery unconfirmed", ambiguous=True
                )
            payload = {}
        if status == 200 and payload.get("ok") is True:
            result = payload.get("result")
            message_id = result.get("message_id") if isinstance(result, dict) else None
            if not isinstance(message_id, int):
                return DeliveryResult(
                    DeliveryStatus.RETRY, "ok=true without a message ID; delivery unconfirmed", ambiguous=True
                )
            return DeliveryResult(DeliveryStatus.SENT, "sent", provider_message_id=str(message_id))
        code = payload.get("error_code") if isinstance(payload.get("error_code"), int) else status
        description = payload.get("description") if isinstance(payload.get("description"), str) else ""
        detail = redact_line(f"HTTP {status}, error {code}: {description or 'no description'}", 200)
        if code == 429:
            parameters = payload.get("parameters")
            retry_after = parameters.get("retry_after") if isinstance(parameters, dict) else None
            if not isinstance(retry_after, (int, float)) or isinstance(retry_after, bool) or retry_after < 0:
                retry_after = None
            return DeliveryResult(DeliveryStatus.RETRY, f"rate limited: {detail}", retry_after_s=retry_after)
        if code in PERMANENT_CODES:
            return DeliveryResult(DeliveryStatus.PERMANENT, detail)
        return DeliveryResult(DeliveryStatus.RETRY, detail)
