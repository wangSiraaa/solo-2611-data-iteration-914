"""Clearing orchestration across the auction service and margin warehouse.

The auction owns the frozen result; this layer drives settlement, exposes
statements, and provides the resume entry point used when a response is
lost.  All settlement writes live in durable receipts, never in memory.
"""

from __future__ import annotations

from typing import Any

from .auction import AuctionService
from .margin import MarginLedger


class ClearingService:
    def __init__(self, db, margin: MarginLedger, auctions: AuctionService):
        self.db = db
        self.margin = margin
        self.auctions = auctions

    def settle(self, request_id: str, auction_id: str) -> dict[str, Any]:
        """Idempotent, resumable settlement.

        Safe to call again with the same request after a timeout or a lost
        response: committed margin receipts are replayed, uncommitted holds
        are processed, and the frozen ranking is read back, never recomputed.
        """
        return self.auctions.settle_auction(request_id, auction_id)

    def account_statement(self, buyer: str) -> dict[str, Any]:
        """Ledger-shaped statement derived from raw events."""
        with self.db.read() as conn:
            events = [
                dict(r)
                for r in conn.execute(
                    "SELECT seq, kind, amount_cents, hold_id, auction_id, request_id, at "
                    "FROM ledger_events WHERE buyer=? ORDER BY seq",
                    (buyer,),
                )
            ]
        totals = {"DEPOSIT": 0, "FREEZE": 0, "RELEASE": 0, "CHARGE": 0}
        for e in events:
            totals[e["kind"]] += e["amount_cents"]
        balance = self.margin.balance(buyer)
        # Event-only derivation:
        #   charged   = sum(CHARGE)
        #   frozen    = FREEZE - RELEASE - CHARGE  (consumed/released holds net to 0)
        #   available = deposited - frozen - charged
        #             = DEPOSIT - FREEZE + RELEASE
        derived = {
            "deposited_cents": totals["DEPOSIT"],
            "frozen_cents": totals["FREEZE"] - totals["RELEASE"] - totals["CHARGE"],
            "charged_cents": totals["CHARGE"],
            "available_cents": totals["DEPOSIT"] - totals["FREEZE"] + totals["RELEASE"],
        }
        return {
            "buyer": buyer,
            "events": events,
            "event_totals": totals,
            "derived": derived,
            "balance": balance,
        }

    def auction_statement(self, auction_id: str) -> dict[str, Any]:
        result = self.auctions.read_result(auction_id)
        with self.db.read() as conn:
            charges = [
                dict(r)
                for r in conn.execute(
                    "SELECT seq, buyer, hold_id, amount_cents, at FROM ledger_events "
                    "WHERE auction_id=? AND kind='CHARGE' ORDER BY seq",
                    (auction_id,),
                )
            ]
            releases = [
                dict(r)
                for r in conn.execute(
                    "SELECT seq, buyer, hold_id, amount_cents, at FROM ledger_events "
                    "WHERE auction_id=? AND kind='RELEASE' ORDER BY seq",
                    (auction_id,),
                )
            ]
        return {
            "auction": result,
            "charge_events": charges,
            "release_events": releases,
            "total_charged_cents": sum(c["amount_cents"] for c in charges),
            "total_released_cents": sum(r["amount_cents"] for r in releases),
        }

    def all_balances(self) -> list[dict[str, Any]]:
        with self.db.read() as conn:
            buyers = [
                r["buyer"]
                for r in conn.execute(
                    "SELECT DISTINCT buyer FROM ledger_events ORDER BY buyer"
                )
            ]
        return [self.margin.balance(b) for b in buyers]
