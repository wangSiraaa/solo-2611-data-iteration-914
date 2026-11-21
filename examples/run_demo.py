"""Re-runnable end-to-end demo: deposit -> bid -> freeze -> settle -> audit.

Run from the repository root:

    python3 -m examples.run_demo            # fresh temporary database
    python3 -m examples.run_demo --db demo.sqlite3

Every step prints real warehouse state, so you can see how margin changes
gate bid acceptance and how the uniform clearing price drives final charges.
All identities and rules are synthetic; nothing touches real payment rails.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

from sealed_auction import AuctionService, ClearingService, Database, MarginLedger
from sealed_auction.core import DomainError, ResponseLost
from sealed_auction.reconcile import reconcile


def show(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def dump(label: str, obj) -> None:
    print(f"\n[{label}]")
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def balances(clearing: ClearingService, buyers) -> None:
    print("\n  margin balances")
    for b in buyers:
        bal = clearing.margin.balance(b)
        print(
            f"    {b}: deposited={bal['deposited_cents']:>6} "
            f"available={bal['available_cents']:>6} "
            f"frozen={bal['frozen_cents']:>6} charged={bal['charged_cents']:>6}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=None, help="SQLite path (default: temp file)")
    parser.add_argument(
        "--run-tag",
        default=None,
        help="namespace suffix; auto-generated for a reused persistent database",
    )
    args = parser.parse_args()

    tmp = None
    if args.db is None:
        tmp = tempfile.TemporaryDirectory()
        db_path = Path(tmp.name) / "demo.sqlite3"
    else:
        db_path = Path(args.db)
    run_tag = args.run_tag or datetime.now().strftime("%H%M%S%f")

    def nid(base: str) -> str:
        return f"{base}-{run_tag}"

    print(f"database: {db_path}  run_tag: {run_tag}")

    db = Database(db_path)
    margin = MarginLedger(db)
    auctions = AuctionService(db, margin)
    clearing = ClearingService(db, margin, auctions)
    # A persistent database may already hold earlier runs; namespace every
    # synthetic account/auction/bid so each run is a clean, auditable episode.
    ephemeral = tmp is not None
    tag = "r1" if ephemeral else run_tag

    def ns(name: str) -> str:
        return name if ephemeral else f"{name}#{tag}"

    buyers = [ns(b) for b in ["alice", "bob", "carol", "dave", "erin"]]
    alice, bob, carol, dave, erin = buyers
    auc_a, auc_b = ns("A"), ns("B")
    counter = {"n": 0}

    def req(tag_name: str) -> str:
        counter["n"] += 1
        return f"demo-{tag}-{tag_name}-{counter['n']:03d}"

    def label(name: str) -> str:
        return name.split("#")[0]

    try:
        show("0. fresh ledger")
        balances(clearing, buyers)

        show("1. deposits (available cash)")
        deposits = {alice: 50_000, bob: 50_000, carol: 50_000,
                    dave: 20_000, erin: 5_000}
        for buyer, amt in deposits.items():
            margin.deposit(req("dep"), buyer, amt)
        balances(clearing, buyers)

        show("2. open auction A (10 units) and place sealed bids")
        auctions.create_auction(req("new"), auc_a, 10)
        # Hold then bid: the auction re-verifies the hold itself.
        def bid(bid_id, buyer, qty, price):
            bid_id = ns(bid_id)
            hold = margin.freeze(req("frz"), buyer, auc_a, qty * price)
            res = auctions.place_bid(
                req("bid"), bid_id, auc_a, buyer, qty, price, hold["hold_id"]
            )
            print(
                f"    {label(bid_id)} {label(buyer):>5} qty={qty} price={price} "
                f"hold={hold['hold_id']} -> accepted rev {res['revision']}"
            )
            return res

        bid("A-01", alice, 4, 500)
        bid("A-02", bob, 4, 500)      # ties A-01 on price, larger id
        bid("A-03", carol, 5, 300)
        bid("A-04", dave, 2, 200)     # will win nothing
        bid("A-05", erin, 3, 100)     # will win nothing
        balances(clearing, buyers)

        show("3a. insufficient collateral: erin tries to raise A-05 3@100 -> 3@2000")
        try:
            margin.freeze(req("frz"), erin, auc_a, 3 * 2_000)
            print("    UNEXPECTED: freeze accepted")
        except DomainError as exc:
            print(f"    freeze rejected: {exc}")

        show("3b. revise carol A-03 5@300 -> 5@350 (atomic hold replacement)")
        new_hold = margin.freeze(req("frz"), carol, auc_a, 5 * 350)
        revised = auctions.revise_bid(
            req("rev"), ns("A-03"), carol, 5, 350, new_hold["hold_id"]
        )
        print(
            f"    revision={revised['revision']} new_hold={revised['hold_id']} "
            f"replaced={revised['replaced_hold_id']} -> "
            f"old status={margin.get_hold(revised['replaced_hold_id'])['status']}"
        )
        balances(clearing, buyers)

        show("4. cutoff (freeze): deterministic ranking, no money moved")
        frozen = auctions.freeze_auction(req("close"), auc_a)
        for a in frozen["allocations"]:
            print(
                f"    rank bid={label(a['bid_id'])} buyer={label(a['buyer']):>5} "
                f"price={a['unit_price_cents']} qty={a['qty']} -> "
                f"allocated={a['allocated_qty']} order={a['order_cents']}"
            )
        print(f"    clearing price = {frozen['clearing_price_cents']} cents (uniform)")
        balances(clearing, buyers)

        show("5. settle: real charges at the uniform price + release remainder")
        settled = clearing.settle(req("settle"), auc_a)
        for s in settled["settlement"]:
            print(
                f"    {label(s['bid_id'])} {label(s['buyer']):>5} "
                f"allocated={s['allocated_qty']} "
                f"charged={s['charged_cents']:>5} hold={s['hold_status']}"
            )
        balances(clearing, buyers)

        show("6. duplicate settle with a fresh request id (no double charge)")
        again = clearing.settle(req("settle"), auc_a)
        print(f"    orders unchanged: {again['orders'] == settled['orders']}")
        balances(clearing, buyers)

        show("7. lost charge response, then resume with the same request id")
        auctions.create_auction(req("new"), auc_b, 2)
        h = margin.freeze(req("frz"), carol, auc_b, 2 * 250)
        auctions.place_bid(
            req("bid"), ns("B-01"), auc_b, carol, 2, 250, h["hold_id"]
        )
        auctions.freeze_auction(req("close"), auc_b)
        fired = {"n": 0}

        def injector(operation, request_id):
            if operation == "margin.charge_capped" and fired["n"] == 0:
                fired["n"] += 1
                return True
            return False

        margin.fault_injector = injector
        rid = req("settle")
        try:
            clearing.settle(rid, auc_b)
        except ResponseLost:
            print("    simulated timeout AFTER charge committed")
        margin.fault_injector = None
        resumed = clearing.settle(rid, auc_b)
        print(f"    resumed: charged={resumed['settlement'][0]['charged_cents']}")
        balances(clearing, buyers)

        show("8. explanation of cutoff")
        explain = auctions.explain_auction(auc_a)
        print(f"    rule: {explain['rule']}")
        print(f"    clearing: {explain['clearing_rule']}")
        for r in explain["ranked"]:
            print(
                f"    {r['rank']}. {label(r['bid_id'])} @ "
                f"{r['unit_price_cents']}: {r['explanation']} [{r['cutoff']}]"
            )

        show("9. accounts from raw deposit/freeze/release/charge events")
        for buyer in buyers:
            st = clearing.account_statement(buyer)
            d = st["derived"]
            ok = d["available_cents"] + d["frozen_cents"] + d["charged_cents"] == d["deposited_cents"]
            print(
                f"    {label(buyer):>5}: deposited={d['deposited_cents']:>6} "
                f"available={d['available_cents']:>6} frozen={d['frozen_cents']:>5} "
                f"charged={d['charged_cents']:>6} conservation={'OK' if ok else 'FAIL'}"
            )

        show("10. independent reconciliation")
        report = reconcile(db)
        report["auctions"] = {
            label(k): {kk: vv for kk, vv in v.items() if kk != "ranked"}
            for k, v in report["auctions"].items()
        }
        report["buyers"] = {label(k): v for k, v in report["buyers"].items()}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ok"] else 1
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
