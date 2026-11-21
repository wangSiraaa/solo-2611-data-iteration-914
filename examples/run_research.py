"""End-to-end research audit: bid menu, one-table test, cheapest hand-off.

Run from the repository root:

    python3 -m examples.run_research

It builds **real** auctions (deposits -> holds -> bids -> cutoff -> settle),
then reconstructs the audit purely from frozen snapshots.  Nothing here
mutates the real settlements or the bank ledger; the only writes go to
research-only tables.

Sections
--------
0.  Worked example from the brief (two 1-unit auctions, no competitors).
1.  A 24-round / 6-category archive built from real frozen auctions, with a
    deliberate contradiction, a protected subset and reviewable fees.
2.  The bid menu: per alternative, signable? units? real spend?
3.  One-table result: a witness table, or the exact contradiction core.
4.  Cheapest manual hand-off (protected rounds excluded) + per-round proof.
5.  Later correction of one round's collateral -> research version only.
6.  Stable-id tie-break turning a partial fill into a full one.
7.  Real settlement and bank balances are byte-for-byte unaffected.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sealed_auction import AuctionService, ClearingService, Database, MarginLedger
from sealed_auction.reconcile import reconcile
from sealed_auction.research import (
    Alternative,
    Archive,
    Round,
    actual_outcome,
    build_bid_menu,
    explain_alternative,
    find_round_core,
    find_valuation_table,
    load_menu,
    load_versions,
    mapping_source,
    merge_sources,
    min_cost_handoff,
    persist_menu,
    snapshot_source,
    verify_table,
)


def show(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def dump(label: str, obj) -> None:
    print(f"\n[{label}]")
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# Minimal engine harness (one database per round: 2 auctions, limit is 3)
# ---------------------------------------------------------------------------


class Engine:
    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "auction.sqlite3")
        self.margin = MarginLedger(self.db)
        self.auctions = AuctionService(self.db, self.margin)
        self.clearing = ClearingService(self.db, self.margin, self.auctions)
        self.n = 0

    def req(self, p="r"):
        self.n += 1
        return f"{p}-{self.n:04d}"

    def deposit(self, buyer, amount):
        self.margin.deposit(self.req("dep"), buyer, amount)

    def build_auction(self, auction_id, units, competitor_bids):
        self.auctions.create_auction(self.req("new"), auction_id, units)
        for comp, qty, price in competitor_bids:
            h = self.margin.freeze(
                self.req("frz"), comp, auction_id, qty * price
            )
            self.auctions.place_bid(
                self.req("bid"), f"{auction_id}-{comp}", auction_id,
                comp, qty, price, h["hold_id"],
            )

    def place_bid(self, auction_id, bid_id, buyer, qty, price):
        h = self.margin.freeze(
            self.req("frz"), buyer, auction_id, qty * price
        )
        return self.auctions.place_bid(
            self.req("bid"), bid_id, auction_id, buyer, qty, price, h["hold_id"]
        )

    def close(self):
        self.auctions.freeze_auction(self.req("close"), self._current)
        # settle is exercised by the caller when desired

    def settle_all(self):
        for aid in self._auction_ids:
            self.auctions.settle_auction(self.req("settle"), aid)

    def snapshot(self, auction_id):
        return snapshot_source(self.db)(auction_id)

    def close_all(self):
        self.tmp.cleanup()


PARTICIPANT = "procurement-participant"


# ---------------------------------------------------------------------------
# 0. Worked example from the brief
# ---------------------------------------------------------------------------


def worked_example() -> None:
    show("0. worked example: two 1-unit auctions, no competitors, range 0..10")

    eng = Engine()
    eng.deposit(PARTICIPANT, 10_000)
    eng.build_auction("U1", 1, [])
    eng._current = "U1"
    eng.place_bid("U1", "U1-A", PARTICIPANT, 1, 4)       # actually submitted A @4
    eng.close()

    eng.deposit(PARTICIPANT, 10_000)
    eng.build_auction("U2", 1, [])
    eng._current = "U2"
    eng.place_bid("U2", "U2-B", PARTICIPANT, 1, 4)       # actually submitted B @4
    eng.close()
    eng.auctions.settle_auction(eng.req("settle"), "U1")
    eng.auctions.settle_auction(eng.req("settle"), "U2")

    src = merge_sources(snapshot_source(eng.db))

    def alt(aid, auction, cat, qty, price, bid_id, coll):
        return Alternative(aid, auction, cat, qty, price, bid_id, coll, PARTICIPANT)

    r1 = Round("round-1",
               [alt("A", "U1", "A", 1, 4, "U1-A", 4),
                alt("B", "U1", "B", 1, 3, "U1-B", 3)],
               chosen_alt="A", review_cost=1)
    r2 = Round("round-2",
               [alt("B", "U2", "B", 1, 4, "U2-B", 4),
                alt("A", "U2", "A", 1, 3, "U2-A", 3)],
               chosen_alt="B", review_cost=4)
    archive = Archive([r1, r2], {"A": (0, 10), "B": (0, 10)})

    print("real outcomes:", actual_outcome(src, "U1", PARTICIPANT),
          actual_outcome(src, "U2", PARTICIPANT))
    for rec in build_bid_menu(archive, src):
        print(
            f"  {rec['round_id']}/{rec['alt_id']}: signable={rec['signable']} "
            f"units={rec['allocated_qty']} spend={rec['spend_cents']}"
        )

    print("\none table for both rounds:", find_valuation_table(archive, src))
    core = find_round_core(archive, src, archive.rounds)
    print("contradiction core:", [r.round_id for r in core])
    print("  round-1 needs A-B >= 1; round-2 needs B-A >= 1 -> impossible")
    ho = min_cost_handoff(archive, src)
    print(f"cheapest hand-off: {ho['handoff_rounds']} fee={ho['total_cost']}")
    print("explaining table after hand-off:", ho["table"])

    revised = archive.revised_copy()
    diff = revised.correct_collateral("round-1", "B", 2)  # alt B never had 3
    print("\ncollateral correction (research copy only):", diff)
    print("round versions now:", revised.versions())
    table = find_valuation_table(revised, src)
    print("one table now:", table)
    print("proof ok:", verify_table(revised, src, table)["ok"])
    eng.close_all()


# ---------------------------------------------------------------------------
# 1. A 24-round / 6-category archive from real auctions
# ---------------------------------------------------------------------------

CATS = ["cat1", "cat2", "cat3", "cat4", "cat5", "cat6"]
# Hidden table consistent with the 22 non-conflicting rounds.
HIDDEN = {"cat1": 4000, "cat2": 5000, "cat3": 6000,
          "cat4": 7000, "cat5": 3000, "cat6": 8000}
BOUNDS = {
    "cat1": (0, 10_000),
    "cat2": (500, 9_000),
    "cat3": (0, 8_000),
    "cat4": (1_000, 10_000),
    "cat5": (0, 6_000),
    "cat6": (2_000, 10_000),
}


def utility(table, alt_eval):
    return alt_eval["allocated_qty"] * table[alt_eval["category"]] - alt_eval["spend_cents"]


def build_archive() -> tuple[Archive, object]:
    """Materialise 24 real rounds (2 auctions each), then audit snapshots.

    Round design (all real auctions; the participant really bids in the
    "chosen" auction and really settles):

    * 22 consistent rounds use only ``cat3..cat6``.  The chosen option always
      has strictly higher utility than the rejected option under ``HIDDEN``;
      construction makes that provable by hand (chosen utility >= 2480, the
      rejected option scores 200, abstention scores 0).
    * rounds ``r22``/``r23`` are the symmetric A/B contradiction: one needs
      ``v[cat1] - v[cat2] >= 1``, the other ``v[cat2] - v[cat1] >= 1``.  The
      22 consistent rounds do not touch ``cat1``/``cat2``, so each conflict
      round is feasible on its own and only the pair collides.
    """
    engs = []
    sources = []

    def new_engine():
        e = Engine()
        engs.append(e)
        sources.append(snapshot_source(e.db))
        return e

    rounds: list[Round] = []
    consistent_cats = ["cat3", "cat4", "cat5", "cat6"]

    for i in range(22):
        c = consistent_cats[i % 4]
        d = consistent_cats[(i + 1) % 4]     # rejected option: different cat
        pc = 100 + 20 * i
        qc = 1 + (i % 3)

        e = new_engine()
        e.deposit(PARTICIPANT, 200_000)
        aid_c = f"R{i:02d}-chosen"
        aid_o = f"R{i:02d}-other"

        # Auction for the chosen option: three shapes exercising full fill,
        # uniform price below own bid, and a partial fill.
        if i % 3 == 0:
            units = qc
            comps = []                                     # full fill @ pc
        elif i % 3 == 1:
            units = qc + 1
            comps = [("comp-low", 1, max(1, pc - 30))]      # price set by rival
        else:
            units = qc
            comps = [("comp-high", 1, pc + 40)]             # partial fill
        for comp, q, p in comps:
            e.deposit(comp, 500_000)
        e.build_auction(aid_c, units, comps)

        # Auction for the rejected option: uncontested single unit.
        e.build_auction(aid_o, 1, [])

        # The participant's one real bid this round goes to the chosen auction.
        e._current = aid_c
        e.place_bid(aid_c, f"{aid_c}-bid", PARTICIPANT, qc, pc)
        e._current = aid_o
        e.close()
        e._current = aid_c
        e.close()
        e._auction_ids = [aid_c, aid_o]

        # Rejected option priced so its counterfactual utility is exactly 200.
        v_d = HIDDEN[d]
        po = v_d - 200

        def alt(aid, tag, cat, qty, price, coll):
            bid = f"{aid}-bid" if tag == "chosen" else f"{aid}-hyp"
            return Alternative(tag, aid, cat, qty, price, bid, coll, PARTICIPANT)

        rounds.append(
            Round(
                f"r{i:02d}",
                [alt(aid_c, "chosen", c, qc, pc, 10_000),
                 alt(aid_o, "other", d, 1, po, 10_000)],
                chosen_alt="chosen",
                protected=(i in (2, 7, 15)),
                review_cost=3 + (i % 4),
            )
        )

    # -- symmetric A/B contradiction (cat1 vs cat2), real uncontested auctions.
    conflict_specs = [
        ("r22", ("cat1", 4), ("cat2", 3), 2),  # pick cat1@4 over cat2@3
        ("r23", ("cat2", 4), ("cat1", 3), 5),  # pick cat2@4 over cat1@3
    ]
    for rid, (cc, pc), (dc, po), fee in conflict_specs:
        e = new_engine()
        e.deposit(PARTICIPANT, 50_000)
        aid_c = f"{rid}-chosen"
        aid_o = f"{rid}-other"
        e.build_auction(aid_c, 1, [])
        e.build_auction(aid_o, 1, [])
        e._current = aid_c
        e.place_bid(aid_c, f"{aid_c}-bid", PARTICIPANT, 1, pc)
        e._current = aid_o
        e.close()
        e._current = aid_c
        e.close()
        e._auction_ids = [aid_c, aid_o]

        def alt(aid, tag, cat, price, coll):
            bid = f"{aid}-bid" if tag == "chosen" else f"{aid}-hyp"
            return Alternative(tag, aid, cat, 1, price, bid, coll, PARTICIPANT)

        rounds.append(
            Round(
                rid,
                [alt(aid_c, "chosen", cc, pc, 4),
                 alt(aid_o, "other", dc, po, 3)],
                chosen_alt="chosen",
                review_cost=fee,
            )
        )

    archive = Archive(rounds, BOUNDS)

    for e in engs:
        e.settle_all()
    src = merge_sources(*sources)
    return archive, (engs, src)


def archive_demo(archive: Archive, src) -> None:
    show("1. archive shape")
    print(f"rounds={len(archive.rounds)} categories={archive.categories()}")
    print(f"bounds={archive.bounds}")
    print("protected:", [r.round_id for r in archive.rounds if r.protected])
    print("reviewable fees:",
          {r.round_id: r.review_cost for r in archive.rounds if r.review_cost is not None})

    show("2. bid menu (selected rows)")
    menu = build_bid_menu(archive, src)
    print(f"menu entries: {len(menu)}")
    for rec in menu:
        if rec["round_id"] in ("r00", "r01", "r22", "r23"):
            print(
                f"  {rec['round_id']}/{rec['alt_id']:8s} "
                f"cat={rec['category']} qty={rec['qty']} bid={rec['unit_price_cents']}\n"
                f"     signable={rec['signable']} units={rec['allocated_qty']} "
                f"uniform={rec['clearing_price_cents']} spend={rec['spend_cents']}"
                + ("" if rec["not_signable_reason"] is None
                   else f"\n     blocked: {rec['not_signable_reason']}")
            )

    show("3. can one valuation table explain everything?")
    table = find_valuation_table(archive, src)
    print("table:", table)
    if table is None:
        core = find_round_core(archive, src, archive.rounds)
        print("contradiction core rounds:", [r.round_id for r in core])
        for r in core:
            print(f"  round {r.round_id} (chosen={r.chosen_alt}):")
            from sealed_auction.research import round_constraints
            for c in round_constraints(archive, src, r):
                print("    -", c["text"])

    show("4. cheapest manual hand-off (protected rounds never removed)")
    ho = min_cost_handoff(archive, src)
    dump("hand-off", {k: ho[k] for k in
                      ("feasible", "handoff_rounds", "total_cost", "table")})
    kept = [r for r in archive.rounds
            if r.round_id not in set(ho["handoff_rounds"])]
    proof = verify_table(archive, src, ho["table"], kept)
    print("per-round proof (kept rounds), all ok =", proof["ok"])
    for chk in proof["rounds"]:
        alts = ", ".join(
            f"{a['alt_id']}:{a['detail']} "
            f"{'<=' if a['chosen_beats'] else '>'} chosen"
            for a in chk["alternatives"]
        )
        print(
            f"  {chk['round_id']}: chosen={chk['chosen_alt']} "
            f"u={chk['chosen_utility']} ({chk['chosen_detail']}) "
            f"abstain_ok={chk['beats_abstain']} | {alts}"
        )
    print("\nrounds removed to manual review:",
          ho["handoff_rounds"], "total fee =", ho["total_cost"])
    print("protected rounds kept:",
          [r.round_id for r in kept if r.protected])

    show("5. later correction: round r22 alt B collateral was never 3")
    revised = archive.revised_copy()
    diff = revised.correct_collateral("r22", "other", 2)
    dump("correction diff", diff)
    print("versions:", revised.versions())
    t2 = find_valuation_table(revised, src)
    print("one table for ALL 24 rounds now:", t2)
    if t2 is not None:
        print("proof:", verify_table(revised, src, t2)["ok"])
    print("frozen original still infeasible:",
          find_valuation_table(archive, src) is None)

    print("\n--- single alternative: why it is not signable / units / spend ---")
    # Round r22's rejected alternative "other" (cat2, 1 @ 3) needs 3 cents but
    # only 2 were available after the collateral correction, so it could not
    # have been signed at all.
    info = explain_alternative(revised, src, "r22", "other")
    print(json.dumps(
        {k: info[k] for k in
         ("round_id", "alt_id", "signable", "not_signable_reason",
          "allocated_qty", "spend_cents", "collateral_cents", "required_cents")},
        indent=2, ensure_ascii=False,
    ))

    show("6. stable bid_id tie-break: partial fill vs full fill")
    from sealed_auction.research import evaluate_alternative
    tie_view = {"units": 2,
                "bids": [{"bid_id": "c-1", "buyer": "comp", "qty": 1,
                          "unit_price_cents": 100}]}
    tie_src = mapping_source({"TIE": tie_view})
    for stable in ("a-01", "zz-9"):
        r = Round("tie", [Alternative("X", "TIE", "cat1", 3, 100, stable,
                                      300, PARTICIPANT)], chosen_alt="X")
        rec = evaluate_alternative(Archive([r], {"cat1": (0, 10)}),
                                   tie_src, r, r.alternatives[0])
        print(
            f"  bid_id={stable:5s} -> allocated={rec['allocated_qty']} of 3, "
            f"spend={rec['spend_cents']} (same price, same rivals)"
        )

    return menu


def persistence_demo(archive, src, menu) -> None:
    show("7. persist menu to research-only tables")
    # Reuse the first engine's database purely as a scratch store.
    engs, _ = archive_holder
    db = engs[0].db
    persist_menu(db, archive, menu)
    print("stored entries:", sum(1 for _ in menu))
    print("single look-up r22/other:",
          load_menu(db, "r22", "other")["allocated_qty"],
          "units at", load_menu(db, "r22", "other")["spend_cents"], "cents")
    print("stored versions:", load_versions(db))


if __name__ == "__main__":
    worked_example()
    archive, (engs, src) = build_archive()
    archive_holder = (engs, src)
    menu = archive_demo(archive, src)

    show("8. real settlement + bank balances are NOT affected")
    # Snapshot real state, run all research again, compare.
    def snap_state():
        state = {"recon": reconcile(engs[0].db)}
        bal = {}
        for e in engs:
            for buyer in [PARTICIPANT, "comp-hi", "comp-lo", "comp-hi2"]:
                try:
                    b = e.margin.balance(buyer)
                    bal[f"{e.db.path}:{buyer}"] = {
                        k: b[k] for k in
                        ("deposited_cents", "frozen_cents", "charged_cents",
                         "available_cents")
                    }
                except Exception:
                    pass
        state["balances"] = bal
        return state

    before = snap_state()
    # Re-run every read-only research operation and a correction copy.
    _ = build_bid_menu(archive, src)
    _ = find_valuation_table(archive, src)
    _ = min_cost_handoff(archive, src)
    rev = archive.revised_copy()
    rev.correct_collateral("r22", "other", 2)
    _ = find_valuation_table(rev, src)
    persist_menu(engs[0].db, archive, build_bid_menu(archive, src))
    after = snap_state()
    print("reconcile ok:", before["recon"]["ok"], "->", after["recon"]["ok"])
    print("balances identical:", before["balances"] == after["balances"])

    persistence_demo(archive, src, menu)

    for e in engs:
        e.close_all()
