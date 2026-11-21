"""Margin warehouse: deposits, holds, releases, capped charges, balances.

Cash accounting is event-sourced in ``ledger_events``; holds are the
materialised per-bid collateral objects.  For every buyer the following
invariant holds at all times:

    total_deposited = available + active_hold_amounts + charged_amounts

``available`` is never stored; it is derived under BEGIN IMMEDIATE so two
connections racing for the last cent cannot both observe it.
"""

from __future__ import annotations

import sqlite3
from typing import Any
from .core import (
    DataIntegrityError,
    DomainError,
    InvalidState,
    NotFound,
    now_utc,
    put_receipt,
    require_int,
    require_nonempty,
)
from .core import get_receipt


class MarginLedger:
    def __init__(self, db, fault_injector=None):
        self.db = db
        # fault_injector(operation, request_id) -> raise ResponseLost after commit
        self.fault_injector = fault_injector

    # ------------------------------------------------------------------ reads

    @staticmethod
    def _legs(conn: sqlite3.Connection, buyer: str) -> dict[str, int]:
        deposited = conn.execute(
            "SELECT COALESCE(SUM(amount_cents),0) AS v FROM ledger_events "
            "WHERE buyer=? AND kind='DEPOSIT'",
            (buyer,),
        ).fetchone()["v"]
        frozen = conn.execute(
            "SELECT COALESCE(SUM(amount_cents),0) AS v FROM holds "
            "WHERE buyer=? AND status='ACTIVE'",
            (buyer,),
        ).fetchone()["v"]
        charged = conn.execute(
            "SELECT COALESCE(SUM(charged_cents),0) AS v FROM holds WHERE buyer=?",
            (buyer,),
        ).fetchone()["v"]
        return {
            "deposited_cents": int(deposited),
            "frozen_cents": int(frozen),
            "charged_cents": int(charged),
            "available_cents": int(deposited) - int(frozen) - int(charged),
        }

    def legs(self, conn: sqlite3.Connection, buyer: str) -> dict[str, int]:
        return self._legs(conn, buyer)

    def balance(self, buyer: str) -> dict[str, Any]:
        require_nonempty(buyer, "buyer")
        with self.db.read() as conn:
            legs = self._legs(conn, buyer)
            holds = [
                dict(r)
                for r in conn.execute(
                    "SELECT hold_id, auction_id, amount_cents, status, charged_cents "
                    "FROM holds WHERE buyer=? ORDER BY created_at, hold_id",
                    (buyer,),
                )
            ]
        return {"buyer": buyer, **legs, "holds": holds}

    def get_hold(self, hold_id: str) -> dict[str, Any]:
        with self.db.read() as conn:
            return self._get_hold_conn(conn, hold_id)

    def _get_hold_conn(self, conn, hold_id: str) -> dict[str, Any]:
        require_nonempty(hold_id, "hold_id")
        row = conn.execute("SELECT * FROM holds WHERE hold_id=?", (hold_id,)).fetchone()
        if row is None:
            raise NotFound(f"hold {hold_id!r} not found")
        return dict(row)

    # --------------------------------------------------------------- writes

    def deposit(
        self, request_id: str, buyer: str, amount_cents: int
    ) -> dict[str, Any]:
        require_nonempty(request_id, "request_id")
        require_nonempty(buyer, "buyer")
        require_int(amount_cents, "amount_cents", minimum=1)
        payload = {"buyer": buyer, "amount_cents": amount_cents}
        operation = "margin.deposit"

        with self.db.tx() as conn:
            prior = get_receipt(conn, request_id, operation, payload)
            if prior is not None:
                return prior
            conn.execute(
                "INSERT INTO ledger_events(buyer, kind, amount_cents, request_id, at) "
                "VALUES (?, 'DEPOSIT', ?, ?, ?)",
                (buyer, amount_cents, request_id, now_utc()),
            )
            legs = self._legs(conn, buyer)
            response = {"request_id": request_id, "buyer": buyer, "amount_cents": amount_cents,
                        **legs}
            put_receipt(conn, request_id, operation, payload, response)
        self._maybe_lose("margin.deposit", request_id, response)
        return response

    def freeze(
        self,
        request_id: str,
        buyer: str,
        auction_id: str,
        amount_cents: int,
        hold_id: str | None = None,
    ) -> dict[str, Any]:
        """Freeze collateral for one auction.  Funds become unavailable to
        every other bid, including another auction."""
        require_nonempty(request_id, "request_id")
        require_nonempty(buyer, "buyer")
        require_nonempty(auction_id, "auction_id")
        require_int(amount_cents, "amount_cents", minimum=1)
        hold_id = hold_id or f"hold-{request_id}"
        require_nonempty(hold_id, "hold_id")
        payload = {"buyer": buyer, "auction_id": auction_id,
                   "amount_cents": amount_cents, "hold_id": hold_id}
        operation = "margin.freeze"

        with self.db.tx() as conn:
            prior = get_receipt(conn, request_id, operation, payload)
            if prior is not None:
                return prior
            legs = self._legs(conn, buyer)
            if legs["available_cents"] < amount_cents:
                raise DomainError(
                    f"insufficient available balance for {buyer}: "
                    f"need {amount_cents}, available {legs['available_cents']} cents"
                )
            at = now_utc()
            try:
                conn.execute(
                    "INSERT INTO holds(hold_id, buyer, auction_id, amount_cents, "
                    "status, charged_cents, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, 'ACTIVE', 0, ?, ?)",
                    (hold_id, buyer, auction_id, amount_cents, at, at),
                )
            except sqlite3.IntegrityError as exc:
                raise DomainError(
                    f"hold {hold_id!r} already exists; use a fresh hold_id"
                ) from exc
            conn.execute(
                "INSERT INTO ledger_events(buyer, kind, amount_cents, hold_id, "
                "auction_id, request_id, at) VALUES (?, 'FREEZE', ?, ?, ?, ?, ?)",
                (buyer, amount_cents, hold_id, auction_id, request_id, at),
            )
            legs = self._legs(conn, buyer)
            response = {
                "request_id": request_id,
                "hold_id": hold_id,
                "buyer": buyer,
                "auction_id": auction_id,
                "amount_cents": amount_cents,
                "status": "ACTIVE",
                **legs,
            }
            put_receipt(conn, request_id, operation, payload, response)
        self._maybe_lose("margin.freeze", request_id, response)
        return response

    def release_in_tx(
        self, conn, hold_id: str, request_id: str, auction_id: str | None = None
    ) -> dict[str, Any]:
        """Release an ACTIVE hold inside a caller-owned transaction.

        Used by bid revision/cancellation so the old collateral disappears in
        the same commit that removes or replaces the bid it backed.
        """
        hold = self._get_hold_conn(conn, hold_id)
        if hold["status"] != "ACTIVE":
            raise InvalidState(
                f"hold {hold_id} is {hold['status']}; only ACTIVE holds can be released"
            )
        if auction_id is not None and hold["auction_id"] != auction_id:
            raise DomainError(
                f"hold {hold_id} belongs to auction {hold['auction_id']}, not {auction_id}"
            )
        at = now_utc()
        conn.execute(
            "UPDATE holds SET status='RELEASED', updated_at=? WHERE hold_id=?",
            (at, hold_id),
        )
        conn.execute(
            "INSERT INTO ledger_events(buyer, kind, amount_cents, hold_id, "
            "auction_id, request_id, at) VALUES (?, 'RELEASE', ?, ?, ?, ?, ?)",
            (hold["buyer"], hold["amount_cents"], hold_id, hold["auction_id"],
             request_id, at),
        )
        return {
            "hold_id": hold_id,
            "buyer": hold["buyer"],
            "released_cents": hold["amount_cents"],
            "status": "RELEASED",
            **self._legs(conn, hold["buyer"]),
        }

    def release(self, request_id: str, hold_id: str) -> dict[str, Any]:
        """Release an ACTIVE hold in full.  Settled holds cannot be released."""
        require_nonempty(request_id, "request_id")
        require_nonempty(hold_id, "hold_id")
        payload = {"hold_id": hold_id}
        operation = "margin.release"

        with self.db.tx() as conn:
            prior = get_receipt(conn, request_id, operation, payload)
            if prior is not None:
                return prior
            hold = self._get_hold_conn(conn, hold_id)
            if hold["status"] != "ACTIVE":
                raise InvalidState(
                    f"hold {hold_id} is {hold['status']}; only ACTIVE holds can be released"
                )
            at = now_utc()
            conn.execute(
                "UPDATE holds SET status='RELEASED', updated_at=? WHERE hold_id=?",
                (at, hold_id),
            )
            conn.execute(
                "INSERT INTO ledger_events(buyer, kind, amount_cents, hold_id, "
                "auction_id, request_id, at) VALUES (?, 'RELEASE', ?, ?, ?, ?, ?)",
                (hold["buyer"], hold["amount_cents"], hold_id, hold["auction_id"],
                 request_id, at),
            )
            legs = self._legs(conn, hold["buyer"])
            response = {
                "request_id": request_id,
                "hold_id": hold_id,
                "buyer": hold["buyer"],
                "released_cents": hold["amount_cents"],
                "status": "RELEASED",
                **legs,
            }
            put_receipt(conn, request_id, operation, payload, response)
        self._maybe_lose("margin.release", request_id, response)
        return response

    def charge_capped(
        self,
        request_id: str,
        hold_id: str,
        auction_id: str,
        charge_cents: int,
    ) -> dict[str, Any]:
        """Consume a hold after settlement.

        The charge is capped at the hold amount.  The unused remainder is
        released in the same transaction.  The hold becomes CONSUMED, so a
        second settlement attempt cannot charge it again.  A database-level
        unique index also forbids two CHARGE events for the same
        (auction, hold).
        """
        require_nonempty(request_id, "request_id")
        require_nonempty(hold_id, "hold_id")
        require_nonempty(auction_id, "auction_id")
        require_int(charge_cents, "charge_cents", minimum=0)
        payload = {"hold_id": hold_id, "auction_id": auction_id,
                   "charge_cents": charge_cents}
        operation = "margin.charge_capped"

        with self.db.tx() as conn:
            prior = get_receipt(conn, request_id, operation, payload)
            if prior is not None:
                return prior
            hold = self._get_hold_conn(conn, hold_id)
            if hold["auction_id"] != auction_id:
                raise DomainError(
                    f"hold {hold_id} belongs to auction {hold['auction_id']}, "
                    f"not {auction_id}"
                )
            if hold["status"] == "CONSUMED" or hold["charged_cents"]:
                raise InvalidState(f"hold {hold_id} has already been settled")
            if hold["status"] != "ACTIVE":
                raise InvalidState(f"hold {hold_id} is {hold['status']}; cannot charge")

            existing = conn.execute(
                "SELECT 1 FROM ledger_events WHERE kind='CHARGE' AND auction_id=? "
                "AND hold_id=?",
                (auction_id, hold_id),
            ).fetchone()
            if existing is not None:
                raise InvalidState("duplicate charge for this auction hold")
            if charge_cents > hold["amount_cents"]:
                raise DataIntegrityError(
                    f"charge {charge_cents} exceeds hold {hold['amount_cents']} "
                    "for hold " + hold_id
                )

            at = now_utc()
            remainder = hold["amount_cents"] - charge_cents
            conn.execute(
                "UPDATE holds SET status='CONSUMED', charged_cents=?, updated_at=? "
                "WHERE hold_id=?",
                (charge_cents, at, hold_id),
            )
            if charge_cents > 0:
                conn.execute(
                    "INSERT INTO ledger_events(buyer, kind, amount_cents, hold_id, "
                    "auction_id, request_id, at) VALUES (?, 'CHARGE', ?, ?, ?, ?, ?)",
                    (hold["buyer"], charge_cents, hold_id, auction_id, request_id, at),
                )
            if remainder > 0:
                conn.execute(
                    "INSERT INTO ledger_events(buyer, kind, amount_cents, hold_id, "
                    "auction_id, request_id, at) VALUES (?, 'RELEASE', ?, ?, ?, ?, ?)",
                    (hold["buyer"], remainder, hold_id, auction_id,
                     request_id + ":remainder", at),
                )
            legs = self._legs(conn, hold["buyer"])
            response = {
                "request_id": request_id,
                "hold_id": hold_id,
                "auction_id": auction_id,
                "buyer": hold["buyer"],
                "charged_cents": charge_cents,
                "released_cents": remainder,
                "status": "CONSUMED",
                **legs,
            }
            put_receipt(conn, request_id, operation, payload, response)
        self._maybe_lose("margin.charge_capped", request_id, response)
        return response

    # ------------------------------------------------------ credential check

    @staticmethod
    def verify_credential(
        conn: sqlite3.Connection,
        hold_id: str,
        buyer: str,
        auction_id: str,
        required_cents: int,
    ) -> dict[str, Any]:
        """Auction-side verification of a presented hold, inside the bid tx.

        The auction never trusts caller-supplied "already frozen" claims: it
        reads the warehouse row and checks owner, auction, amount and status.
        """
        row = conn.execute("SELECT * FROM holds WHERE hold_id=?", (hold_id,)).fetchone()
        if row is None:
            raise DomainError(f"unknown hold credential {hold_id!r}")
        hold = dict(row)
        errors = []
        if hold["buyer"] != buyer:
            errors.append(f"buyer mismatch ({hold['buyer']} != {buyer})")
        if hold["auction_id"] != auction_id:
            errors.append(f"auction mismatch ({hold['auction_id']} != {auction_id})")
        if hold["status"] != "ACTIVE":
            errors.append(f"hold status is {hold['status']}")
        if hold["amount_cents"] < required_cents:
            errors.append(
                f"hold {hold['amount_cents']} < required {required_cents} cents"
            )
        if errors:
            raise DomainError("invalid hold credential: " + "; ".join(errors))
        return hold

    # ------------------------------------------------------------- fault hook

    def _maybe_lose(self, operation: str, request_id: str, response: dict) -> None:
        if self.fault_injector is not None:
            from .core import ResponseLost

            if self.fault_injector(operation, request_id):
                raise ResponseLost(operation, request_id, response)
