"""Shared errors, time helpers and durable idempotency receipts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


class DomainError(Exception):
    """Base class for expected domain-level rejections."""


class InsufficientFunds(DomainError):
    pass


class InsufficientHold(DomainError):
    pass


class InvalidCredential(DomainError):
    pass


class InvalidState(DomainError):
    pass


class NotFound(DomainError):
    pass


class LimitExceeded(DomainError):
    pass


class IdempotencyConflict(DomainError):
    pass


class DataIntegrityError(DomainError):
    pass


class ResponseLost(Exception):
    """Transport-level failure after the database transaction committed.

    The caller must retry with the same request_id.  This exception is only
    raised by the fault-injection hooks used in tests/demo.
    """

    def __init__(self, operation: str, request_id: str, response: dict[str, Any]):
        super().__init__(
            f"response lost after commit: operation={operation} request_id={request_id}"
        )
        self.operation = operation
        self.request_id = request_id
        self.response = response


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def require_int(value: Any, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DomainError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise DomainError(f"{name} must be >= {minimum}, got {value}")
    return value


def require_nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DomainError(f"{name} must be a non-empty string")
    return value


def payload_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_receipt(
    conn, request_id: str, operation: str, payload: dict[str, Any]
) -> dict[str, Any] | None:
    """Return a previously committed response for this request, if any.

    A reused request_id with a different operation or payload is a hard error:
    silently replaying a different request would corrupt the ledger.
    """
    row = conn.execute(
        "SELECT operation, fingerprint, response_json FROM receipts WHERE request_id=?",
        (request_id,),
    ).fetchone()
    if row is None:
        return None
    if row["operation"] != operation or row["fingerprint"] != payload_fingerprint(payload):
        raise IdempotencyConflict(
            f"request_id {request_id!r} already used for a different request"
        )
    return json.loads(row["response_json"])


def put_receipt(
    conn,
    request_id: str,
    operation: str,
    payload: dict[str, Any],
    response: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO receipts(request_id, operation, fingerprint, response_json, at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            request_id,
            operation,
            payload_fingerprint(payload),
            json.dumps(response, sort_keys=True, ensure_ascii=False),
            now_utc(),
        ),
    )
