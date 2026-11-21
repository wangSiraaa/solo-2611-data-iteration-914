"""End-to-end scenario tests for the sealed auction + margin warehouse."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from sealed_auction import AuctionService, ClearingService, Database, MarginLedger
from sealed_auction.core import (
    DomainError,
    InvalidState,
    LimitExceeded,
    ResponseLost,
)
from sealed_auction.reconcile import assert_reconciled, reconcile


class Harness:
    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self._tmp.name) / "auction.sqlite3")
        self.margin = MarginLedger(self.db)
        self.auctions = AuctionService(self.db, self.margin)
        self.clearing = ClearingService(self.db, self.margin, self.auctions)
        self._req = 0

    def req(self, prefix="req"):
        self._req += 1
        return f"{prefix}-{self._req:04d}"

    def deposit(self, buyer, amount):
        return self.margin.deposit(self.req("dep"), buyer, amount)

    def create(self, auction_id, units):
        return self.auctions.create_auction(self.req("new"), auction_id, units)

    def freeze_hold(self, buyer, auction_id, amount, hold_id=None):
        return self.margin.freeze(
            self.req("frz"), buyer, auction_id, amount, hold_id=hold_id
        )

    def bid(self, bid_id, auction_id, buyer, qty, price, hold_id=None):
        hold = self.freeze_hold(buyer, auction_id, qty * price, hold_id=hold_id)
        return self.auctions.place_bid(
            self.req("bid"), bid_id, auction_id, buyer, qty, price, hold["hold_id"]
        )

    def close(self):
        self._tmp.cleanup()


class ScenarioTest(unittest.TestCase):
    def setUp(self):
        self.h = Harness()

    def tearDown(self):
        self.h.close()

    # --------------------------------------------------------------- 1. tie

    def test_tie_truncation_uses_bid_id_not_input_order(self):
        h = self.h
        for b in ("b1", "b2", "b3"):
            h.deposit(b, 10_000)
        h.create("a-tie", 7)
        # b2 places first but has the lexicographically larger bid id.
        h.bid("t-02", "a-tie", "b2", 4, 500)
        h.bid("t-01", "a-tie", "b1", 4, 500)
        h.bid("t-03", "a-tie", "b3", 3, 300)

        frozen = h.auctions.freeze_auction(h.req("close"), "a-tie")
        by_id = {a["bid_id"]: a for a in frozen["allocations"]}
        self.assertEqual(by_id["t-01"]["allocated_qty"], 4)
        self.assertEqual(by_id["t-02"]["allocated_qty"], 3)  # partial, loses tie
        self.assertEqual(by_id["t-03"]["allocated_qty"], 0)
        self.assertEqual(frozen["clearing_price_cents"], 500)
        self.assertEqual(frozen["units_allocated"], 7)

        result = h.clearing.settle(h.req("settle"), "a-tie")
        self.assertEqual(result["clearing_price_cents"], 500)
        # Input order (t-02 first) did not decide the truncation.
        self.assertNotEqual(
            [o["buyer"] for o in result["orders"]], ["b2", "b1"]
        )
        assert_reconciled(h.db)

    # ------------------------------------------------- 2. uniform price/partial

    def test_uniform_price_is_lowest_allocated_and_orders_use_it(self):
        h = self.h
        for b in ("b1", "b2", "b3", "b4", "b5"):
            h.deposit(b, 20_000)
        h.create("a-price", 10)
        h.bid("p-01", "a-price", "b1", 4, 500)
        h.bid("p-02", "a-price", "b2", 4, 500)
        h.bid("p-03", "a-price", "b3", 5, 300)
        h.bid("p-04", "a-price", "b4", 2, 200)
        h.bid("p-05", "a-price", "b5", 3, 100)

        frozen = h.auctions.freeze_auction(h.req("close"), "a-price")
        self.assertEqual(frozen["clearing_price_cents"], 300)
        by_id = {a["bid_id"]: a for a in frozen["allocations"]}
        self.assertEqual(by_id["p-01"]["allocated_qty"], 4)
        self.assertEqual(by_id["p-02"]["allocated_qty"], 4)
        self.assertEqual(by_id["p-03"]["allocated_qty"], 2)  # last bid partial
        self.assertEqual(by_id["p-04"]["allocated_qty"], 0)
        self.assertEqual(by_id["p-05"]["allocated_qty"], 0)
        # Every winner pays the same 300, even the 500 bidders.
        self.assertEqual(
            {o["order_cents"] for o in frozen["orders"]}, {1200, 600}
        )
        self.assertEqual(sum(o["allocated_qty"] for o in frozen["orders"]), 10)

        settled = h.clearing.settle(h.req("settle"), "a-price")
        b1 = h.margin.balance("b1")
        b3 = h.margin.balance("b3")
        b4 = h.margin.balance("b4")
        self.assertEqual(b1["charged_cents"], 1200)  # 4 * 300, not 4 * 500
        self.assertEqual(b3["charged_cents"], 600)   # 2 * 300
        self.assertEqual(b4["charged_cents"], 0)
        self.assertEqual(b4["available_cents"], 20_000)  # fully released
        # Holds never exceed bid notional and charge never exceeds hold.
        for item in settled["settlement"]:
            self.assertLessEqual(item["charged_cents"], item["allocated_qty"] * 500)
        assert_reconciled(h.db)

    # ------------------------------------------------------ 3. revise/cancel

    def test_revise_releases_old_hold_atomically_and_uses_new_price(self):
        h = self.h
        h.deposit("b1", 5_000)
        h.create("a-rev", 3)
        first = h.bid("r-01", "a-rev", "b1", 2, 300)  # hold 600
        old_hold = first["hold_id"]
        new_hold = h.freeze_hold("b1", "a-rev", 2 * 500)["hold_id"]

        revised = h.auctions.revise_bid(
            h.req("rev"), "r-01", "b1", 2, 500, new_hold
        )
        self.assertEqual(revised["revision"], 2)
        self.assertEqual(h.margin.get_hold(old_hold)["status"], "RELEASED")
        self.assertEqual(h.margin.get_hold(new_hold)["status"], "ACTIVE")

        frozen = h.auctions.freeze_auction(h.req("close"), "a-rev")
        self.assertEqual(frozen["clearing_price_cents"], 500)
        self.assertEqual(frozen["allocations"][0]["hold_id"], new_hold)

        # Cancel now releases the replacement hold and removes the bid.
        h.create("a-rev2", 3)
        b2 = h.bid("r-02", "a-rev2", "b1", 1, 100)  # hold 100
        h.auctions.cancel_bid(h.req("cancel"), "r-02")
        self.assertEqual(h.margin.get_hold(b2["hold_id"])["status"], "RELEASED")
        frozen2 = h.auctions.freeze_auction(h.req("close2"), "a-rev2")
        self.assertEqual(frozen2["units_allocated"], 0)
        self.assertIsNone(frozen2["clearing_price_cents"])
        assert_reconciled(h.db)

    # -------------------------------------------------------- 4. insufficient

    def test_insufficient_funds_and_credentials(self):
        h = self.h
        h.deposit("b1", 1_000)
        h.create("a-x", 3)

        # Freeze for more than available is rejected.
        with self.assertRaises(DomainError):
            h.margin.freeze(h.req("frz"), "b1", "a-x", 1_001)

        # Hold exists but does not cover qty * price: auction rejects the bid.
        hold = h.freeze_hold("b1", "a-x", 1_000)["hold_id"]
        with self.assertRaises(DomainError):
            h.auctions.place_bid(h.req("bid"), "x-01", "a-x", "b1", 3, 400, hold)

        # A caller cannot assert "already frozen" with a made-up credential.
        with self.assertRaises(DomainError):
            h.auctions.place_bid(
                h.req("bid"), "x-02", "a-x", "b1", 1, 100, "hold-does-not-exist"
            )

        # Credential bound to another buyer is rejected.
        h.deposit("b2", 1_000)
        hold_b2 = h.freeze_hold("b2", "a-x", 100)["hold_id"]
        with self.assertRaises(DomainError):
            h.auctions.place_bid(h.req("bid"), "x-03", "a-x", "b1", 1, 100, hold_b2)

        # Released credentials are rejected.
        h.margin.release(h.req("rel"), hold)
        with self.assertRaises(DomainError):
            h.auctions.place_bid(h.req("bid"), "x-04", "a-x", "b1", 1, 100, hold)

        self.assertEqual(reconcile(h.db)["ok"], True)

    def test_hold_cannot_back_two_auctions(self):
        h = self.h
        h.deposit("b1", 1_000)
        h.create("a-1", 2)
        h.create("a-2", 2)
        hold = h.freeze_hold("b1", "a-1", 900)["hold_id"]
        with self.assertRaises(DomainError):
            # Same credential pointed at a different auction.
            h.auctions.place_bid(h.req("bid"), "c-02", "a-2", "b1", 3, 300, hold)

    # ------------------------------------------------------- 5. duplicate settle

    def test_duplicate_settlement_charges_once(self):
        h = self.h
        h.deposit("b1", 10_000)
        h.create("a-dup", 2)
        h.bid("d-01", "a-dup", "b1", 2, 400)
        h.auctions.freeze_auction(h.req("close"), "a-dup")

        rid = h.req("settle")
        first = h.clearing.settle(rid, "a-dup")
        second = h.clearing.settle(rid, "a-dup")
        third = h.clearing.settle(h.req("settle"), "a-dup")  # new request id
        self.assertEqual(first["orders"], second["orders"])
        self.assertEqual(first["orders"], third["orders"])

        with h.db.read() as conn:
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM ledger_events "
                "WHERE auction_id='a-dup' AND kind='CHARGE'"
            ).fetchone()["n"]
        self.assertEqual(n, 1)
        self.assertEqual(h.margin.balance("b1")["charged_cents"], 800)
        assert_reconciled(h.db)

    # ---------------------------------------------- 6. lost charge response

    def test_lost_charge_response_is_resumed_not_double_charged(self):
        h = self.h
        h.deposit("b1", 10_000)
        h.deposit("b2", 10_000)
        h.create("a-lost", 4)
        h.bid("l-01", "a-lost", "b1", 2, 400)
        h.bid("l-02", "a-lost", "b2", 2, 300)
        h.auctions.freeze_auction(h.req("close"), "a-lost")

        fired = {"count": 0}

        def injector(operation, request_id):
            if operation == "margin.charge_capped" and fired["count"] == 0:
                fired["count"] += 1
                return True
            return False

        h.margin.fault_injector = injector
        rid = h.req("settle")
        with self.assertRaises(ResponseLost):
            h.clearing.settle(rid, "a-lost")
        # The first charge committed; the rest never ran.  A timeout is not
        # proof of failure, so we retry with the same request id.
        h.margin.fault_injector = None
        resumed = h.clearing.settle(rid, "a-lost")
        self.assertEqual(resumed["clearing_price_cents"], 300)

        with h.db.read() as conn:
            rows = conn.execute(
                "SELECT hold_id, COUNT(*) AS n FROM ledger_events "
                "WHERE auction_id='a-lost' AND kind='CHARGE' GROUP BY hold_id"
            ).fetchall()
        self.assertTrue(all(r["n"] == 1 for r in rows))
        # price 300: b1 2*300=600, b2 2*300=600
        self.assertEqual(h.margin.balance("b1")["charged_cents"], 600)
        self.assertEqual(h.margin.balance("b2")["charged_cents"], 600)
        assert_reconciled(h.db)

    def test_deposit_lost_response_is_idempotent(self):
        h = self.h
        fired = {"done": False}

        def injector(operation, request_id):
            if operation == "margin.deposit" and not fired["done"]:
                fired["done"] = True
                return True
            return False

        h.margin.fault_injector = injector
        rid = h.req("dep")
        with self.assertRaises(ResponseLost):
            h.margin.deposit(rid, "b1", 1_000)
        h.margin.fault_injector = None
        result = h.margin.deposit(rid, "b1", 1_000)
        self.assertEqual(result["deposited_cents"], 1_000)  # not 2,000
        assert_reconciled(h.db)

    # --------------------------------------------- 7. immutability after cutoff

    def test_frozen_result_is_immutable_and_read_back_identically(self):
        h = self.h
        h.deposit("b1", 5_000)
        h.deposit("b2", 5_000)
        h.create("a-fix", 3)
        h.bid("f-01", "a-fix", "b1", 2, 400)
        h.bid("f-02", "a-fix", "b2", 2, 200)
        frozen = h.auctions.freeze_auction(h.req("close"), "a-fix")

        # No bid mutations after cutoff.
        with self.assertRaises(InvalidState):
            h.auctions.place_bid(
                h.req("bid"), "f-03", "a-fix", "b2", 1, 100,
                h.freeze_hold("b2", "a-fix", 100)["hold_id"],
            )
        with self.assertRaises(InvalidState):
            h.auctions.cancel_bid(h.req("cancel"), "f-01")
        new_hold = h.freeze_hold("b1", "a-fix", 100)["hold_id"]
        with self.assertRaises(InvalidState):
            h.auctions.revise_bid(h.req("rev"), "f-01", "b1", 1, 100, new_hold)

        # A charge failure cannot rewrite the ranked result.
        h.margin.fault_injector = lambda op, rid: op == "margin.charge_capped"
        with self.assertRaises(ResponseLost):
            h.clearing.settle(h.req("settle"), "a-fix")
        h.margin.fault_injector = None
        reopened = h.auctions.read_result("a-fix")
        self.assertEqual(reopened["units_allocated"], frozen["units_allocated"])
        self.assertEqual(
            reopened["clearing_price_cents"], frozen["clearing_price_cents"]
        )
        self.assertEqual(
            [a["bid_id"] for a in reopened["allocations"]],
            [a["bid_id"] for a in frozen["allocations"]],
        )
        self.assertEqual(
            [a["allocated_qty"] for a in reopened["allocations"]],
            [a["allocated_qty"] for a in frozen["allocations"]],
        )
        # Resume completes without re-ranking.
        h.clearing.settle(h.req("settle2"), "a-fix")
        final = h.auctions.read_result("a-fix")
        self.assertEqual(
            [a["allocated_qty"] for a in final["allocations"]],
            [a["allocated_qty"] for a in frozen["allocations"]],
        )
        assert_reconciled(h.db)

    # -------------------------------------------------- 8. concurrency last cent

    def test_two_connections_racing_for_last_cent(self):
        h = self.h
        h.deposit("b1", 1_000)
        results = []
        barrier = threading.Barrier(2)

        def worker(auction_id, amount):
            barrier.wait()
            try:
                h.margin.freeze(
                    f"race-{auction_id}", "b1", auction_id, amount
                )
                results.append(("ok", auction_id))
            except DomainError as exc:
                results.append(("rejected", str(exc)))

        threads = [
            threading.Thread(target=worker, args=("a-race-1", 1_000)),
            threading.Thread(target=worker, args=("a-race-2", 1_000)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        successes = [r for r in results if r[0] == "ok"]
        rejections = [r for r in results if r[0] == "rejected"]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(rejections), 1)
        self.assertEqual(h.margin.balance("b1")["frozen_cents"], 1_000)
        self.assertEqual(h.margin.balance("b1")["available_cents"], 0)
        assert_reconciled(h.db)

    # ------------------------------------------------------- 9. edge limits

    def test_empty_auction_has_no_clearing_price(self):
        h = self.h
        h.create("a-empty", 5)
        frozen = h.auctions.freeze_auction(h.req("close"), "a-empty")
        self.assertIsNone(frozen["clearing_price_cents"])
        self.assertEqual(frozen["units_allocated"], 0)
        result = h.clearing.settle(h.req("settle"), "a-empty")
        self.assertIsNone(result["clearing_price_cents"])
        assert_reconciled(h.db)

    def test_limits_and_one_bid_per_buyer(self):
        h = self.h
        for i in range(5):
            h.deposit(f"b{i}", 10_000)
        with self.assertRaises(LimitExceeded):
            h.auctions.create_auction(h.req("new"), "a-big", 11)

        h.create("a-lim", 2)
        h.create("a-lim-2", 2)
        h.create("a-lim-3", 2)
        with self.assertRaises(LimitExceeded):
            h.create("a-lim-4", 2)
        h.bid("m-00", "a-lim", "b0", 1, 100)
        with self.assertRaises(InvalidState):
            # Same buyer, second active bid (credential is valid, bid is not).
            dup_hold = h.freeze_hold("b0", "a-lim", 100)["hold_id"]
            h.auctions.place_bid(
                h.req("bid"), "m-00b", "a-lim", "b0", 1, 100, dup_hold
            )

        h.bid("m-01", "a-lim", "b1", 1, 100)
        h.bid("m-02", "a-lim", "b2", 1, 100)
        h.bid("m-03", "a-lim", "b3", 1, 100)
        h.bid("m-04", "a-lim", "b4", 1, 100)
        h.deposit("b5", 10_000)
        with self.assertRaises(LimitExceeded):
            h.bid("m-05", "a-lim", "b5", 1, 100)

    # ------------------------------------------------------ 10. statements

    def test_statements_and_explanation_reconcile_with_events(self):
        h = self.h
        h.deposit("b1", 5_000)
        h.create("a-exp", 4)
        h.bid("e-01", "a-exp", "b1", 3, 250)
        h.auctions.freeze_auction(h.req("close"), "a-exp")
        h.clearing.settle(h.req("settle"), "a-exp")

        statement = h.clearing.account_statement("b1")
        self.assertEqual(statement["derived"]["deposited_cents"], 5_000)
        self.assertEqual(statement["derived"]["charged_cents"], 750)
        self.assertEqual(statement["derived"]["frozen_cents"], 0)
        self.assertEqual(statement["derived"]["available_cents"], 4_250)
        self.assertEqual(
            statement["balance"]["charged_cents"], 750
        )

        explain = h.auctions.explain_auction("a-exp")
        self.assertEqual(explain["clearing_price_cents"], 250)
        self.assertEqual(
            [e["action"] for e in explain["bid_events"]], ["PLACE"]
        )
        self.assertEqual(explain["ranked"][0]["order_cents"], 750)

        auction_statement = h.clearing.auction_statement("a-exp")
        self.assertEqual(auction_statement["total_charged_cents"], 750)
        self.assertEqual(auction_statement["total_released_cents"], 0)
        assert_reconciled(h.db)

    # ------------------------------------------------- 11. tamper detection

    def test_reconcile_detects_a_deleted_charge(self):
        h = self.h
        h.deposit("b1", 5_000)
        h.create("a-tamper", 2)
        h.bid("k-01", "a-tamper", "b1", 2, 100)
        h.auctions.freeze_auction(h.req("close"), "a-tamper")
        h.clearing.settle(h.req("settle"), "a-tamper")
        self.assertTrue(reconcile(h.db)["ok"])

        with h.db.tx() as conn:
            conn.execute(
                "DELETE FROM ledger_events WHERE auction_id='a-tamper' "
                "AND kind='CHARGE'"
            )
        report = reconcile(h.db)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("CHARGE events" in v or "CONSUMED" in v for v in report["violations"]),
            report["violations"],
        )


    # ------------------------------------------------- 12. concurrent settle

    def test_concurrent_settlement_charges_once(self):
        h = self.h
        h.deposit("b1", 10_000)
        h.create("a-conc", 2)
        h.bid("c-01", "a-conc", "b1", 2, 300)
        h.auctions.freeze_auction(h.req("close"), "a-conc")

        errors = []
        barrier = threading.Barrier(2)

        def worker(i):
            barrier.wait()
            try:
                h.clearing.settle(f"settle-conc-{i}", "a-conc")
            except Exception as exc:  # pragma: no cover - surfaced via errors list
                errors.append(repr(exc))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        with h.db.read() as conn:
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM ledger_events "
                "WHERE auction_id='a-conc' AND kind='CHARGE'"
            ).fetchone()["n"]
        self.assertEqual(n, 1)
        self.assertEqual(h.margin.balance("b1")["charged_cents"], 600)
        assert_reconciled(h.db)


    def test_partial_crash_resumed_by_new_request_id(self):
        """Settlement request ids are only for the outer call; the warehouse
        per-hold request ids are deterministic.  A crash after some charges
        can therefore be resumed even by a caller that lost its request id."""
        h = self.h
        h.deposit("b1", 10_000)
        h.deposit("b2", 10_000)
        h.create("a-crash", 4)
        h.bid("q-01", "a-crash", "b1", 2, 400)
        h.bid("q-02", "a-crash", "b2", 2, 400)
        h.auctions.freeze_auction(h.req("close"), "a-crash")

        fired = {"count": 0}

        def injector(operation, request_id):
            if operation == "margin.charge_capped" and fired["count"] < 1:
                fired["count"] += 1
                return True
            return False

        h.margin.fault_injector = injector
        with self.assertRaises(ResponseLost):
            h.clearing.settle(h.req("settle"), "a-crash")
        h.margin.fault_injector = None
        # Different outer request id, but the same frozen result and the same
        # deterministic per-hold warehouse request ids.
        result = h.clearing.settle(h.req("settle"), "a-crash")
        self.assertEqual(result["clearing_price_cents"], 400)
        with h.db.read() as conn:
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM ledger_events "
                "WHERE auction_id='a-crash' AND kind='CHARGE'"
            ).fetchone()["n"]
        self.assertEqual(n, 2)  # one per winning host, never two
        self.assertEqual(h.margin.balance("b1")["charged_cents"], 800)
        self.assertEqual(h.margin.balance("b2")["charged_cents"], 800)
        assert_reconciled(h.db)


if __name__ == "__main__":
    unittest.main(verbosity=2)
