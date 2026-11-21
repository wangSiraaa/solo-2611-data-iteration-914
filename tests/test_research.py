"""Tests for the research/audit layer and its exact feasibility solver."""

from __future__ import annotations

import itertools
import random
import tempfile
import unittest
from pathlib import Path

from sealed_auction import AuctionService, ClearingService, Database, MarginLedger
from sealed_auction.ilp import integer_feasible
from sealed_auction.reconcile import reconcile
from sealed_auction.research import (
    Alternative,
    Archive,
    Round,
    _allocate,
    actual_outcome,
    build_bid_menu,
    evaluate_alternative,
    explain_alternative,
    find_round_core,
    find_valuation_table,
    load_menu,
    load_versions,
    mapping_source,
    min_cost_handoff,
    persist_menu,
    snapshot_source,
    verify_table,
)

P = "participant"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class Engine:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "a.sqlite3")
        self.margin = MarginLedger(self.db)
        self.auctions = AuctionService(self.db, self.margin)
        self.clearing = ClearingService(self.db, self.margin, self.auctions)
        self.n = 0

    def req(self, p="r"):
        self.n += 1
        return f"{p}-{self.n:04d}"

    def deposit(self, b, amt):
        self.margin.deposit(self.req("dep"), b, amt)

    def create(self, aid, units):
        self.auctions.create_auction(self.req("new"), aid, units)

    def bid(self, aid, bid_id, buyer, qty, price):
        h = self.margin.freeze(self.req("frz"), buyer, aid, qty * price)
        return self.auctions.place_bid(
            self.req("bid"), bid_id, aid, buyer, qty, price, h["hold_id"]
        )

    def close(self, aid):
        return self.auctions.freeze_auction(self.req("close"), aid)

    def settle(self, aid):
        return self.auctions.settle_auction(self.req("settle"), aid)

    def close_all(self):
        self.tmp.cleanup()


def alt(alt_id, auction_id, category, qty, price, bid_id, collateral, buyer=P):
    return Alternative(
        alt_id, auction_id, category, qty, price, bid_id, collateral, buyer
    )


def mapping_archive(views, rounds, bounds=None):
    return Archive(rounds, bounds or {}), mapping_source(views)


# ---------------------------------------------------------------------------
# 1. exact solver
# ---------------------------------------------------------------------------


class SolverTest(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = random.Random(7)
        for _ in range(1500):
            n = rng.randint(1, 3)
            hi = [rng.randint(0, 5) for _ in range(n)]
            lo = [rng.randint(0, h) for h in hi]
            rows = []
            for _ in range(rng.randint(0, 5)):
                rows.append(
                    (
                        [rng.randint(-3, 3) for _ in range(n)],
                        rng.choice(["<=", ">=", "=="]),
                        rng.randint(-5, 9),
                    )
                )
            got = integer_feasible(n, rows, lo, hi)
            exp = None
            for x in itertools.product(*[range(lo[j], hi[j] + 1) for j in range(n)]):
                if all(
                    (s == "<=" and sum(c * v for c, v in zip(co, x)) <= r)
                    or (s == ">=" and sum(c * v for c, v in zip(co, x)) >= r)
                    or (s == "==" and sum(c * v for c, v in zip(co, x)) == r)
                    for co, s, r in rows
                ):
                    exp = list(x)
                    break
            self.assertEqual(got is None, exp is None, (rows, lo, hi, got, exp))
            if got is not None:
                for co, s, r in rows:
                    total = sum(c * v for c, v in zip(co, got))
                    self.assertTrue(
                        (s == "<=" and total <= r)
                        or (s == ">=" and total >= r)
                        or (s == "==" and total == r)
                    )

    def test_handles_valuation_ceiling(self):
        # Two categories up to 10_000: a bound this large must still terminate.
        table = integer_feasible(
            2,
            [([1, 0], ">=", 10000), ([1, -1], ">=", 9999)],
            [0, 0],
            [10000, 10000],
        )
        self.assertIsNotNone(table)
        self.assertEqual(table[0], 10000)
        self.assertGreaterEqual(table[0] - table[1], 9999)


# ---------------------------------------------------------------------------
# 2. counterfactual engine
# ---------------------------------------------------------------------------


class CounterfactualTest(unittest.TestCase):
    def test_reclears_uniform_price_instead_of_reusing_today_price(self):
        # Two competitors: one high (rival-high), one low (rival-low).
        # Participant's actual bid was 200 -> price 200.  The alternative at
        # 400 would raise both winners' price; the alternative at 150 would
        # change the allocation.  Today's 200 must not be reused.
        view = {
            "units": 3,
            "bids": [
                {"bid_id": "h", "buyer": "rival-hi", "qty": 1, "unit_price_cents": 500},
                {"bid_id": "l", "buyer": "rival-lo", "qty": 1, "unit_price_cents": 100},
            ],
        }
        archive, src = mapping_archive(
            {"A": view},
            [Round("r", [alt("x", "A", "cat", 2, 400, "p-x", 800)], chosen_alt="x")],
            {"cat": (0, 10000)},
        )
        rec = evaluate_alternative(archive, src, archive.rounds[0],
                                   archive.rounds[0].alternatives[0])
        # Priority: 500 (1) -> 400 (our 2, but only 2 units left) -> 100 out.
        self.assertEqual(rec["allocated_qty"], 2)
        self.assertEqual(rec["clearing_price_cents"], 400)  # not today's 200
        self.assertEqual(rec["spend_cents"], 800)

    def test_partial_fill_at_uniform_price(self):
        view = {
            "units": 2,
            "bids": [
                {"bid_id": "a", "buyer": "rival", "qty": 1, "unit_price_cents": 500},
            ],
        }
        archive, src = mapping_archive(
            {"A": view},
            [Round("r", [alt("x", "A", "cat", 3, 300, "p-x", 900)], chosen_alt="x")],
            {"cat": (0, 10000)},
        )
        rec = evaluate_alternative(archive, src, archive.rounds[0],
                                   archive.rounds[0].alternatives[0])
        self.assertEqual(rec["allocated_qty"], 1)      # only one unit left
        self.assertEqual(rec["clearing_price_cents"], 300)
        self.assertEqual(rec["spend_cents"], 300)

    def test_alternatives_are_not_stacked_and_auctions_are_not_shared(self):
        # Same units, same eventual total, but different competitors: the
        # alternative feasible set must come from *its own* auction.
        v1 = {"units": 2, "bids": [
            {"bid_id": "c", "buyer": "rival", "qty": 2, "unit_price_cents": 500}]}
        v2 = {"units": 2, "bids": []}
        rounds = [
            Round("r", [alt("x1", "A1", "cat", 2, 300, "b1", 600),
                        alt("x2", "A2", "cat", 2, 300, "b2", 600)], chosen_alt="x1"),
        ]
        archive, src = mapping_archive({"A1": v1, "A2": v2}, rounds,
                                       {"cat": (0, 10000)})
        m1 = evaluate_alternative(archive, src, rounds[0], rounds[0].alternatives[0])
        m2 = evaluate_alternative(archive, src, rounds[0], rounds[0].alternatives[1])
        self.assertEqual(m1["allocated_qty"], 0)  # outbid by the 500 rival
        self.assertEqual(m2["allocated_qty"], 2)  # uncontested
        # Only one alternative is ever inserted: count participant bids.
        self.assertEqual(sum(b["buyer"] == P for b in m2["competitor_bids"]), 0)

    def test_stable_bid_id_turns_full_fill_into_partial_fill(self):
        view = {"units": 2, "bids": [
            {"bid_id": "c-1", "buyer": "rival", "qty": 1, "unit_price_cents": 100}]}
        res = {}
        for stable in ("a-01", "zz-9"):
            r = Round("t", [alt("x", "T", "cat", 3, 100, stable, 300)],
                      chosen_alt="x")
            archive, src = mapping_archive({"T": view}, [r], {"cat": (0, 10000)})
            rec = evaluate_alternative(archive, src, r, r.alternatives[0])
            res[stable] = (rec["allocated_qty"], rec["spend_cents"])
        self.assertEqual(res["a-01"], (2, 200))   # wins the tie, full remainder
        self.assertEqual(res["zz-9"], (1, 100))   # loses the tie, one unit only

    def test_insufficient_collateral_is_not_signable_and_reported(self):
        # The rejected alternative (not the chosen one) cannot be signed.
        rounds = [Round("r", [alt("x", "A", "cat", 1, 100, "b-x", 100),
                              alt("y", "A", "cat", 3, 100, "b-y", 250)],
                        chosen_alt="x")]
        archive, src = mapping_archive(
            {"A": {"units": 3, "bids": []}}, rounds, {"cat": (0, 10000)}
        )
        rec = explain_alternative(archive, src, "r", "y")
        self.assertFalse(rec["signable"])
        self.assertIn("300", rec["not_signable_reason"])
        self.assertIn("250", rec["not_signable_reason"])
        # A non-signable alternative contributes no utility constraint.
        from sealed_auction.research import round_constraints
        texts = [c["text"] for c in round_constraints(archive, src, rounds[0])]
        self.assertFalse(any("b-y" in t or "3*" in t for t in texts))


# ---------------------------------------------------------------------------
# 3. one-table explanation, cores, hand-off
# ---------------------------------------------------------------------------


class TableTest(unittest.TestCase):
    def _ab_views(self):
        return {
            "U1": {"units": 1, "bids": []},
            "U2": {"units": 1, "bids": []},
        }

    def _ab_archive(self, r1_chosen="A", r2_chosen="B"):
        r1 = Round(
            "round-1",
            [alt("A", "U1", "A", 1, 4, "U1-A", 4),
             alt("B", "U1", "B", 1, 3, "U1-B", 3)],
            chosen_alt=r1_chosen, review_cost=1,
        )
        r2 = Round(
            "round-2",
            [alt("B", "U2", "B", 1, 4, "U2-B", 4),
             alt("A", "U2", "A", 1, 3, "U2-A", 3)],
            chosen_alt=r2_chosen, review_cost=4,
        )
        return Archive([r1, r2], {"A": (0, 10), "B": (0, 10)})

    def test_worked_example_contradiction_core_and_handoff(self):
        archive = self._ab_archive()
        src = mapping_source(self._ab_views())
        self.assertIsNone(find_valuation_table(archive, src))
        core = find_round_core(archive, src, archive.rounds)
        self.assertEqual({r.round_id for r in core}, {"round-1", "round-2"})

        ho = min_cost_handoff(archive, src)
        self.assertTrue(ho["feasible"])
        self.assertEqual(ho["handoff_rounds"], ["round-1"])  # cheaper (fee 1)
        self.assertEqual(ho["total_cost"], 1)
        self.assertTrue(ho["verification"]["ok"])

    def test_each_round_alone_is_feasible(self):
        archive = self._ab_archive()
        src = mapping_source(self._ab_views())
        for r in archive.rounds:
            self.assertIsNotNone(find_valuation_table(archive, src, [r]))

    def test_collateral_correction_changes_only_research_version(self):
        archive = self._ab_archive()
        src = mapping_source(self._ab_views())
        before = archive.versions()
        revised = archive.revised_copy()
        diff = revised.correct_collateral("round-1", "B", 2)
        self.assertEqual(diff["other_round_versions"], {"round-2": 1})
        self.assertEqual(revised.versions(), {"round-1": 2, "round-2": 1})
        self.assertEqual(before, {"round-1": 1, "round-2": 1})
        # Original stays infeasible; the corrected copy admits A=4, B=5.
        self.assertIsNone(find_valuation_table(archive, src))
        table = find_valuation_table(revised, src)
        self.assertEqual(table, {"A": 4, "B": 5})
        self.assertTrue(verify_table(revised, src, table)["ok"])

    def test_protected_round_is_never_handed_off(self):
        # Make the cheap round protected; the hand-off must pay the expensive
        # alternative instead.
        archive = self._ab_archive()
        archive.round("round-1").protected = True
        src = mapping_source(self._ab_views())
        ho = min_cost_handoff(archive, src)
        self.assertEqual(ho["handoff_rounds"], ["round-2"])
        self.assertEqual(ho["total_cost"], 4)

    def test_all_protected_contradiction_is_unfixable(self):
        archive = self._ab_archive()
        archive.round("round-1").protected = True
        archive.round("round-2").protected = True
        src = mapping_source(self._ab_views())
        ho = min_cost_handoff(archive, src)
        self.assertFalse(ho["feasible"])
        self.assertIn("protected", ho["reason"])

    def test_consistent_rounds_need_no_handoff(self):
        archive = self._ab_archive(r1_chosen="A", r2_chosen="A")
        # Round 2 now picks A@3 over B@4: requires A - B >= 1 and A >= 3.
        src = mapping_source(self._ab_views())
        table = find_valuation_table(archive, src)
        self.assertIsNotNone(table)
        self.assertGreaterEqual(table["A"] - table["B"], 1)
        ho = min_cost_handoff(archive, src)
        self.assertEqual(ho["handoff_rounds"], [])
        self.assertEqual(ho["total_cost"], 0)

    def test_report_identifies_the_offending_choices(self):
        from sealed_auction.research import round_constraints
        archive = self._ab_archive()
        src = mapping_source(self._ab_views())
        texts = {
            r.round_id: [c["text"] for c in round_constraints(archive, src, r)]
            for r in archive.rounds
        }
        self.assertTrue(any(">= 4 - 3" in t for t in texts["round-1"]))
        self.assertTrue(any(">= 4 - 3" in t for t in texts["round-2"]))
        ho = min_cost_handoff(archive, src)
        self.assertIn(["round-1", "round-2"], ho["conflict_cores"])


# ---------------------------------------------------------------------------
# 4. real engine integration + read-only guarantee
# ---------------------------------------------------------------------------


class EngineIntegrationTest(unittest.TestCase):
    def test_audit_from_real_snapshots_leaves_settlement_and_ledger_intact(self):
        eng = Engine()
        try:
            eng.deposit(P, 100_000)
            eng.deposit("rival", 100_000)
            eng.create("AU1", 2)
            eng.create("AU2", 1)
            eng.bid("AU1", "rival-1", "rival", 1, 300)
            eng.bid("AU1", "p-1", P, 2, 250)      # partial? 1 unit left
            eng.bid("AU2", "p-2", P, 1, 400)
            eng.close("AU1")
            eng.close("AU2")
            eng.settle("AU1")
            eng.settle("AU2")

            src = snapshot_source(eng.db)
            actual1 = actual_outcome(src, "AU1", P)
            self.assertEqual(actual1["allocated_qty"], 1)
            self.assertEqual(actual1["clearing_price_cents"], 250)

            rounds = [
                Round("g1", [alt("p-1", "AU1", "catX", 2, 250, "p-1", 500),
                             alt("p-alt", "AU1", "catY", 2, 200, "p-alt", 400)],
                      chosen_alt="p-1", review_cost=1),
                Round("g2", [alt("p-2", "AU2", "catX", 1, 400, "p-2", 400),
                             alt("p-2b", "AU2", "catY", 1, 350, "p-2b", 350)],
                      chosen_alt="p-2", review_cost=2),
            ]
            archive = Archive(rounds, {"catX": (0, 10000), "catY": (0, 10000)})

            before = reconcile(eng.db)
            bal_before = {
                b: {k: eng.margin.balance(b)[k] for k in
                    ("deposited_cents", "frozen_cents", "charged_cents",
                     "available_cents")}
                for b in (P, "rival")
            }
            menu = build_bid_menu(archive, src)
            _ = find_valuation_table(archive, src)
            _ = min_cost_handoff(archive, src)
            revised = archive.revised_copy()
            revised.correct_collateral("g1", "p-alt", 100)
            _ = find_valuation_table(revised, src)
            persist_menu(eng.db, archive, menu)

            after = reconcile(eng.db)
            bal_after = {
                b: {k: eng.margin.balance(b)[k] for k in
                    ("deposited_cents", "frozen_cents", "charged_cents",
                     "available_cents")}
                for b in (P, "rival")
            }
            self.assertTrue(before["ok"])
            self.assertTrue(after["ok"])
            self.assertEqual(bal_before, bal_after)
            self.assertEqual(before["auctions"], after["auctions"])
        finally:
            eng.close_all()

    def test_research_menu_persists_and_is_queryable(self):
        eng = Engine()
        try:
            eng.deposit(P, 10_000)
            eng.create("A", 3)
            eng.bid("A", "p-1", P, 2, 100)
            eng.close("A")
            eng.settle("A")
            src = snapshot_source(eng.db)
            rounds = [Round("r", [alt("x", "A", "cat", 2, 100, "p-1", 200)],
                            chosen_alt="x", review_cost=1)]
            archive = Archive(rounds, {"cat": (0, 10000)})
            persist_menu(eng.db, archive, build_bid_menu(archive, src))
            rec = load_menu(eng.db, "r", "x")
            self.assertEqual(rec["allocated_qty"], 2)
            self.assertEqual(rec["spend_cents"], 200)
            self.assertEqual(load_versions(eng.db), {"r": 1})
        finally:
            eng.close_all()

    def test_allocate_matches_engine_cutoff(self):
        # Direct rule cross-check: price desc, bid_id asc, partial last.
        bids = [
            {"bid_id": "b", "buyer": "x", "qty": 4, "unit_price_cents": 500},
            {"bid_id": "a", "buyer": "y", "qty": 4, "unit_price_cents": 500},
            {"bid_id": "c", "buyer": "z", "qty": 3, "unit_price_cents": 300},
        ]
        alloc, price = _allocate(7, bids)
        self.assertEqual(alloc, {"a": 4, "b": 3, "c": 0})
        self.assertEqual(price, 500)


if __name__ == "__main__":
    unittest.main(verbosity=2)
