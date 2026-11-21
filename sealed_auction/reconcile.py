"""Independent reconciliation from raw events.

This module deliberately does not call the service write paths.  It rebuilds
balances and auction totals from the smallest raw records:

  * ledger_events  (DEPOSIT / FREEZE / RELEASE / CHARGE)
  * bids + bid_events (final valid bid set)
  * settlements.allocations_json (frozen result)

and reports every discrepancy instead of trusting cached balances.
"""

from __future__ import annotations

import json
from typing import Any

from .core import DataIntegrityError


def reconcile(db) -> dict[str, Any]:
    violations: list[str] = []
    with db.read() as conn:
        buyers = [
            r["buyer"]
            for r in conn.execute(
                "SELECT DISTINCT buyer FROM ledger_events ORDER BY buyer"
            )
        ]

        # ---------------- per-buyer balances from raw events ----------------
        buyer_reports: dict[str, Any] = {}
        for buyer in buyers:
            legs = {"DEPOSIT": 0, "FREEZE": 0, "RELEASE": 0, "CHARGE": 0}
            for r in conn.execute(
                "SELECT kind, amount_cents FROM ledger_events WHERE buyer=?",
                (buyer,),
            ):
                legs[r["kind"]] += r["amount_cents"]
            available = legs["DEPOSIT"] - legs["FREEZE"] + legs["RELEASE"]
            frozen = legs["FREEZE"] - legs["RELEASE"] - legs["CHARGE"]
            charged = legs["CHARGE"]
            accounted = available + frozen + charged
            if accounted != legs["DEPOSIT"]:
                violations.append(
                    f"{buyer}: available+frozen+charged ({accounted}) != "
                    f"deposited ({legs['DEPOSIT']})"
                )
            if available < 0 or frozen < 0:
                violations.append(
                    f"{buyer}: negative available ({available}) or frozen ({frozen})"
                )

            # Each hold's money must split exactly into release + charge legs.
            for hold in conn.execute(
                "SELECT hold_id, amount_cents, status, charged_cents FROM holds "
                "WHERE buyer=?",
                (buyer,),
            ):
                legs_by_kind = {"RELEASE": 0, "CHARGE": 0, "FREEZE": 0}
                for r in conn.execute(
                    "SELECT kind, amount_cents FROM ledger_events WHERE hold_id=?",
                    (hold["hold_id"],),
                ):
                    legs_by_kind[r["kind"]] += r["amount_cents"]
                if legs_by_kind["FREEZE"] != hold["amount_cents"]:
                    violations.append(
                        f"hold {hold['hold_id']}: FREEZE events "
                        f"{legs_by_kind['FREEZE']} != hold amount {hold['amount_cents']}"
                    )
                charged_legs = legs_by_kind["CHARGE"]
                release_legs = legs_by_kind["RELEASE"]
                if charged_legs != hold["charged_cents"]:
                    violations.append(
                        f"hold {hold['hold_id']}: hold.charged {hold['charged_cents']} "
                        f"!= CHARGE events {charged_legs}"
                    )
                # Money legs must match the hold's lifecycle state.
                if hold["status"] == "ACTIVE":
                    if charged_legs or release_legs:
                        violations.append(
                            f"hold {hold['hold_id']}: ACTIVE but already has "
                            f"terminal legs (charge={charged_legs}, "
                            f"release={release_legs})"
                        )
                elif hold["status"] == "RELEASED":
                    if charged_legs:
                        violations.append(
                            f"hold {hold['hold_id']}: RELEASED but charged "
                            f"{charged_legs}"
                        )
                    if release_legs != hold["amount_cents"]:
                        violations.append(
                            f"hold {hold['hold_id']}: RELEASED but release "
                            f"{release_legs} != amount {hold['amount_cents']}"
                        )
                elif hold["status"] == "CONSUMED":
                    if charged_legs + release_legs != hold["amount_cents"]:
                        violations.append(
                            f"hold {hold['hold_id']}: CONSUMED but charge "
                            f"{charged_legs} + release {release_legs} != "
                            f"amount {hold['amount_cents']}"
                        )
                else:
                    violations.append(
                        f"hold {hold['hold_id']}: unknown status {hold['status']}"
                    )

            buyer_reports[buyer] = {
                "deposited_cents": legs["DEPOSIT"],
                "frozen_cents": frozen,
                "charged_cents": charged,
                "available_cents": available,
                "sum_equals_deposits": accounted == legs["DEPOSIT"],
            }

        # ---------------- per-auction ranking + totals ----------------
        auction_reports: dict[str, Any] = {}
        for auction in conn.execute(
            "SELECT auction_id, units, status FROM auctions ORDER BY auction_id"
        ):
            aid = auction["auction_id"]
            row = conn.execute(
                "SELECT clearing_price_cents, allocations_json, status "
                "FROM settlements WHERE auction_id=?",
                (aid,),
            ).fetchone()
            charges = conn.execute(
                "SELECT COALESCE(SUM(amount_cents),0) AS v FROM ledger_events "
                "WHERE auction_id=? AND kind='CHARGE'",
                (aid,),
            ).fetchone()["v"]
            if row is None:
                if auction["status"] != "OPEN":
                    violations.append(f"{aid}: non-open auction without settlement")
                auction_reports[aid] = {"status": auction["status"], "settled": False}
                continue

            snap = json.loads(row["allocations_json"])
            alloc = snap["allocations"]
            # Independent re-derivation from the final valid bid set.
            valid = [
                dict(r)
                for r in conn.execute(
                    "SELECT bid_id, buyer, qty, unit_price_cents, revision, hold_id "
                    "FROM bids WHERE auction_id=? AND status='ACTIVE' "
                    "ORDER BY unit_price_cents DESC, bid_id ASC",
                    (aid,),
                )
            ]
            remaining = auction["units"]
            expected = []
            for b in valid:
                take = min(b["qty"], remaining) if remaining > 0 else 0
                remaining -= take
                expected.append((b, take))
            exp_price = None
            for b, take in expected:
                if take > 0:
                    exp_price = b["unit_price_cents"]
            actual_map = {a["bid_id"]: a for a in alloc}
            expected_ids = {b["bid_id"] for b, _ in expected}
            if expected_ids != set(actual_map):
                violations.append(f"{aid}: frozen bid set differs from final valid bids")
            for b, take in expected:
                a = actual_map.get(b["bid_id"])
                if a is None:
                    continue
                if a["allocated_qty"] != take:
                    violations.append(
                        f"{aid}/{b['bid_id']}: frozen qty {a['allocated_qty']} != "
                        f"re-derived {take}"
                    )
                if a["hold_id"] != b["hold_id"]:
                    violations.append(
                        f"{aid}/{b['bid_id']}: hold credential mismatch in freeze"
                    )
            if snap["clearing_price_cents"] != exp_price:
                violations.append(
                    f"{aid}: clearing price {snap['clearing_price_cents']} != "
                    f"re-derived {exp_price}"
                )
            price = snap["clearing_price_cents"]
            expected_charge_total = 0
            for a in alloc:
                if a["allocated_qty"] > 0 and price is not None:
                    if a["order_cents"] != a["allocated_qty"] * price:
                        violations.append(
                            f"{aid}/{a['bid_id']}: order {a['order_cents']} != "
                            f"qty*price {a['allocated_qty'] * price}"
                        )
                    expected_charge_total += a["allocated_qty"] * price
            if snap["units_allocated"] != auction["units"] - remaining:
                violations.append(
                    f"{aid}: units_allocated {snap['units_allocated']} != "
                    f"{auction['units'] - remaining}"
                )
            if snap["units_allocated"] > auction["units"]:
                violations.append(f"{aid}: allocated more units than offered")
            # Charges from raw events must equal the settled order total once CLEARED.
            if row["status"] == "CLEARED" and charges != expected_charge_total:
                violations.append(
                    f"{aid}: raw CHARGE total {charges} != expected orders "
                    f"{expected_charge_total}"
                )
            dup = conn.execute(
                "SELECT hold_id, COUNT(*) AS n FROM ledger_events WHERE auction_id=? "
                "AND kind='CHARGE' GROUP BY hold_id HAVING n > 1",
                (aid,),
            ).fetchall()
            if dup:
                violations.append(f"{aid}: duplicate CHARGE for holds {[d[0] for d in dup]}")
            # Every valid bid's hold reached a terminal state after clearing.
            if row["status"] == "CLEARED":
                for b in valid:
                    h = conn.execute(
                        "SELECT status FROM holds WHERE hold_id=?", (b["hold_id"],)
                    ).fetchone()
                    if h is None or h["status"] not in ("CONSUMED", "RELEASED"):
                        violations.append(
                            f"{aid}/{b['bid_id']}: hold not terminal after clearing"
                        )

            auction_reports[aid] = {
                "status": auction["status"],
                "settlement_status": row["status"],
                "units": auction["units"],
                "units_allocated": snap["units_allocated"],
                "clearing_price_cents": snap["clearing_price_cents"],
                "orders_total_cents": expected_charge_total,
                "raw_charge_total_cents": charges,
                "ranked": [
                    {
                        "bid_id": a["bid_id"],
                        "buyer": a["buyer"],
                        "price": a["unit_price_cents"],
                        "allocated_qty": a["allocated_qty"],
                        "order_cents": a["order_cents"],
                    }
                    for a in alloc
                ],
            }

    return {
        "ok": not violations,
        "violations": violations,
        "buyers": buyer_reports,
        "auctions": auction_reports,
    }


def assert_reconciled(db) -> dict[str, Any]:
    report = reconcile(db)
    if not report["ok"]:
        raise DataIntegrityError(
            "reconciliation failed:\n" + "\n".join(report["violations"])
        )
    return report
