"""Research / audit layer over frozen sealed-auction archives.

This module answers three questions about a participant's *pre-registered
choice set* without ever touching the real settlement or the margin ledger:

1. **Bid menu (counterfactual).**  For every alternative in a round, if the
   participant had submitted that one alternative instead of the bid they
   actually submitted, how many units would they really win and what would
   they really pay?  The answer is recomputed from the *other buyers'* frozen
   bids using the same rules as :meth:`AuctionService._compute_result`
   (price DESC, stable ``bid_id`` ASC, partial fill on the last unit, uniform
   price = lowest allocated price).  Today's clearing price is never reused:
   inserting a bid can move the uniform price, so the whole auction is
   re-cleared.

2. **One valuation table.**  The procurement officer claims a single integer
   table (per-unit value per exhibition category, inside a declared range)
   rationalises every choice: the chosen alternative maximises
   ``sum(allocated units * value) - payment`` and abstaining scores zero.
   We either return such a table or an infeasible *round core* — the exact
   set of choices that cannot coexist.

3. **Cheapest manual hand-off.**  If some reviewable rounds may be handed to
   humans (protected rounds never), find a minimum-total-fee set of rounds to
   remove so the remainder is explained by one table.

Counterfactual evaluation is a pure function of stored snapshots.  Nothing
here writes to ``auctions``, ``bids``, ``holds`` or ``ledger_events``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

from .ilp import Constraint, integer_feasible

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Alternative:
    """One pre-registered option in a round.

    ``collateral_cents`` is the effective collateral the participant could
    actually put behind *this* alternative at the time (only one of the
    round's alternatives is ever submitted, so these are never stacked).
    ``bid_id`` is the participant's stable id for this alternative; it decides
    same-price ties.
    """

    alt_id: str
    auction_id: str
    category: str
    qty: int
    unit_price_cents: int
    bid_id: str
    collateral_cents: int
    buyer: str

    def required_cents(self) -> int:
        return self.qty * self.unit_price_cents

    def to_dict(self) -> dict:
        return {
            "alt_id": self.alt_id,
            "auction_id": self.auction_id,
            "category": self.category,
            "qty": self.qty,
            "unit_price_cents": self.unit_price_cents,
            "bid_id": self.bid_id,
            "collateral_cents": self.collateral_cents,
            "buyer": self.buyer,
            "required_cents": self.required_cents(),
        }


@dataclass
class Round:
    round_id: str
    alternatives: list[Alternative]
    chosen_alt: str | None = None          # None == abstained
    protected: bool = False                # protected rounds may never be handed off
    review_cost: int | None = None         # fee if reviewable; None == not reviewable
    version: int = 1

    def alternative(self, alt_id: str) -> Alternative:
        for a in self.alternatives:
            if a.alt_id == alt_id:
                return a
        raise KeyError(f"round {self.round_id!r} has no alternative {alt_id!r}")

    def to_dict(self) -> dict:
        return {
            "round_id": self.round_id,
            "chosen_alt": self.chosen_alt,
            "protected": self.protected,
            "review_cost": self.review_cost,
            "version": self.version,
            "alternatives": [a.to_dict() for a in self.alternatives],
        }


@dataclass
class Archive:
    """A participant's frozen choice witness plus research metadata.

    ``bounds`` maps category -> (lo, hi): the declared integer valuation
    interval per unit (0 <= lo <= hi, hi <= 10000 in supplied material).
    """

    rounds: list[Round]
    bounds: dict[str, tuple[int, int]]
    note: str = ""

    # -- lookups ---------------------------------------------------------
    def round(self, round_id: str) -> Round:
        for r in self.rounds:
            if r.round_id == round_id:
                return r
        raise KeyError(round_id)

    def categories(self) -> list[str]:
        cats = set(self.bounds)
        for r in self.rounds:
            for a in r.alternatives:
                cats.add(a.category)
        return sorted(cats)

    def revised_copy(self) -> "Archive":
        """Deep copy so an audit revision leaves the frozen original intact."""
        return Archive(
            rounds=[
                replace(r, alternatives=[replace(a) for a in r.alternatives])
                for r in self.rounds
            ],
            bounds=dict(self.bounds),
            note=self.note,
        )

    # -- later correction of one round's collateral material -------------
    def correct_collateral(
        self, round_id: str, alt_id: str, collateral_cents: int
    ) -> dict:
        """Correct only this round's research version.

        Returns a small diff so the caller can verify no other round moved.
        """
        r = self.round(round_id)
        alt = r.alternative(alt_id)
        before = alt.collateral_cents
        alt.collateral_cents = collateral_cents
        r.version += 1
        return {
            "round_id": round_id,
            "alt_id": alt_id,
            "collateral_cents_before": before,
            "collateral_cents_after": collateral_cents,
            "round_version": r.version,
            "other_round_versions": {
                rr.round_id: rr.version for rr in self.rounds if rr.round_id != round_id
            },
        }

    def versions(self) -> dict[str, int]:
        return {r.round_id: r.version for r in self.rounds}


# ---------------------------------------------------------------------------
# Frozen auction snapshots + deterministic re-clearing
# ---------------------------------------------------------------------------

Bid = dict  # keys: bid_id, buyer, qty, unit_price_cents


def _allocate(units: int, bids: Sequence[Bid]) -> tuple[dict[str, int], int | None]:
    """Replicate the auction cutoff exactly (read-only)."""
    ordered = sorted(bids, key=lambda b: (-b["unit_price_cents"], b["bid_id"]))
    remaining = units
    alloc: dict[str, int] = {}
    clearing: int | None = None
    for b in ordered:
        take = min(b["qty"], remaining) if remaining > 0 else 0
        remaining -= take
        alloc[b["bid_id"]] = take
        if take > 0:
            clearing = b["unit_price_cents"]
    return alloc, clearing


def snapshot_source(db) -> Callable[[str], dict]:
    """Build ``auction_id -> {units, bids}`` from frozen settlement snapshots."""

    def get(auction_id: str) -> dict:
        with db.read() as conn:
            row = conn.execute(
                "SELECT s.allocations_json, a.units FROM settlements s "
                "JOIN auctions a ON a.auction_id = s.auction_id "
                "WHERE s.auction_id=?",
                (auction_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"auction {auction_id!r} has no frozen result")
        snap = json.loads(row["allocations_json"])
        bids = [
            {
                "bid_id": a["bid_id"],
                "buyer": a["buyer"],
                "qty": a["qty"],
                "unit_price_cents": a["unit_price_cents"],
            }
            for a in snap["allocations"]
        ]
        return {"units": int(row["units"]), "bids": bids}

    return get


def mapping_source(views: Mapping[str, dict]) -> Callable[[str], dict]:
    def get(auction_id: str) -> dict:
        if auction_id not in views:
            raise KeyError(f"no snapshot for auction {auction_id!r}")
        return views[auction_id]

    return get


def merge_sources(*sources: Callable[[str], dict]) -> Callable[[str], dict]:
    cache: dict[str, dict] = {}

    def get(auction_id: str) -> dict:
        if auction_id in cache:
            return cache[auction_id]
        for src in sources:
            try:
                cache[auction_id] = src(auction_id)
                return cache[auction_id]
            except KeyError:
                continue
        raise KeyError(auction_id)

    return get


# ---------------------------------------------------------------------------
# Counterfactual evaluation
# ---------------------------------------------------------------------------


def evaluate_alternative(
    archive: Archive,
    source: Callable[[str], dict],
    round: Round,
    alt: Alternative,
) -> dict:
    """Re-clear ``alt.auction_id`` with only this alternative inserted."""
    view = source(alt.auction_id)
    # Replace the participant's own bid in that auction (never stack alternatives).
    competitors = [b for b in view["bids"] if b["buyer"] != alt.buyer]
    bids = competitors + [
        {
            "bid_id": alt.bid_id,
            "buyer": alt.buyer,
            "qty": alt.qty,
            "unit_price_cents": alt.unit_price_cents,
        }
    ]
    alloc, clearing = _allocate(view["units"], bids)
    got = alloc.get(alt.bid_id, 0)
    spend = got * clearing if clearing is not None else 0
    required = alt.required_cents()
    signable = required <= alt.collateral_cents
    chosen = alt.alt_id == round.chosen_alt
    return {
        "round_id": round.round_id,
        "alt_id": alt.alt_id,
        "auction_id": alt.auction_id,
        "category": alt.category,
        "bid_id": alt.bid_id,
        "qty": alt.qty,
        "unit_price_cents": alt.unit_price_cents,
        "required_cents": required,
        "collateral_cents": alt.collateral_cents,
        "signable": signable or chosen,  # the chosen one really was submitted
        "chosen": chosen,
        "allocated_qty": got,
        "clearing_price_cents": clearing,
        "spend_cents": spend,
        "units_offered": view["units"],
        "competitor_bids": sorted(
            competitors, key=lambda b: (-b["unit_price_cents"], b["bid_id"])
        ),
        "not_signable_reason": (
            None
            if (signable or chosen)
            else f"needs {required} cents of collateral but only "
            f"{alt.collateral_cents} was available"
        ),
    }


def build_bid_menu(
    archive: Archive, source: Callable[[str], dict]
) -> list[dict]:
    """Auditable menu: every round, every alternative, counterfactual outcome."""
    out = []
    for r in archive.rounds:
        for a in r.alternatives:
            out.append(evaluate_alternative(archive, source, r, a))
    return out


def explain_alternative(
    archive: Archive, source: Callable[[str], dict], round_id: str, alt_id: str
) -> dict:
    """Single-alternative view: signable? units? spend? and why."""
    r = archive.round(round_id)
    alt = r.alternative(alt_id)
    rec = evaluate_alternative(archive, source, r, alt)
    rec["read_only"] = True
    return rec


def actual_outcome(
    source: Callable[[str], dict], auction_id: str, participant: str
) -> dict:
    """Real allocation/payment of the participant in a frozen auction."""
    view = source(auction_id)
    alloc, clearing = _allocate(view["units"], view["bids"])
    # Sum over this participant's bids (there is at most one per auction).
    got = 0
    ids = []
    for b in view["bids"]:
        if b["buyer"] == participant:
            ids.append(b["bid_id"])
            got += alloc.get(b["bid_id"], 0)
    spend = got * clearing if clearing is not None else 0
    return {
        "auction_id": auction_id,
        "buyer": participant,
        "bid_ids": ids,
        "allocated_qty": got,
        "clearing_price_cents": clearing,
        "spend_cents": spend,
    }


# ---------------------------------------------------------------------------
# One-table consistency
# ---------------------------------------------------------------------------


def _cat_vec(cats: list[str], values: Mapping[str, int]) -> list[int]:
    return [values.get(c, 0) for c in cats]


def _row(cats: list[str], terms: Mapping[str, int], sense: str, rhs: int) -> Constraint:
    vec = [0] * len(cats)
    for c, v in terms.items():
        vec[cats.index(c)] += v
    return (vec, sense, rhs)


def round_constraints(
    archive: Archive,
    source: Callable[[str], dict],
    round: Round,
) -> list[dict]:
    """Linear utility constraints this round imposes on the valuation table.

    Each returned dict carries the wire constraint plus a readable form so an
    auditor can see exactly which comparison caused trouble.
    """
    cats = archive.categories()
    evaluated = [evaluate_alternative(archive, source, round, a) for a in round.alternatives]
    by_id = {e["alt_id"]: e for e in evaluated}
    legal = [e for e in evaluated if e["signable"]]
    rows: list[dict] = []

    if round.chosen_alt is None:
        # Abstention must be at least as good as every signable alternative.
        for e in legal:
            rows.append(
                {
                    "row": _row(cats, {e["category"]: -e["allocated_qty"]}, ">=",
                                -e["spend_cents"]),
                    "text": f"abstain(0) >= {e['alt_id']}: "
                    f"-{e['allocated_qty']}*v[{e['category']}] >= -{e['spend_cents']}",
                }
            )
        return rows

    chosen = by_id[round.chosen_alt]
    cc, cq, cs = chosen["category"], chosen["allocated_qty"], chosen["spend_cents"]
    # Chosen vs abstain.
    rows.append(
        {
            "row": _row(cats, {cc: cq}, ">=", cs),
            "text": f"chosen {chosen['alt_id']} >= abstain: "
            f"{cq}*v[{cc}] >= {cs}",
        }
    )
    for e in legal:
        if e["alt_id"] == round.chosen_alt:
            continue
        terms = {cc: cq}
        terms[e["category"]] = terms.get(e["category"], 0) - e["allocated_qty"]
        rhs = cs - e["spend_cents"]
        rows.append(
            {
                "row": _row(cats, terms, ">=", rhs),
                "text": f"chosen {chosen['alt_id']} >= {e['alt_id']}: "
                f"{cq}*v[{cc}] - {e['allocated_qty']}*v[{e['category']}] "
                f">= {cs} - {e['spend_cents']}",
            }
        )
    return rows


def _bounds_lists(archive: Archive):
    cats = archive.categories()
    lo, hi = [], []
    for c in cats:
        low, high = archive.bounds.get(c, (0, 10000))
        lo.append(low)
        hi.append(high)
    return cats, lo, hi


def find_valuation_table(
    archive: Archive,
    source: Callable[[str], dict],
    rounds: Sequence[Round] | None = None,
) -> dict | None:
    """One integer table explaining every supplied round, or ``None``."""
    use = list(rounds) if rounds is not None else archive.rounds
    cats, lo, hi = _bounds_lists(archive)
    rows: list[Constraint] = []
    for r in use:
        rows.extend(c["row"] for c in round_constraints(archive, source, r))
    point = integer_feasible(len(cats), rows, lo, hi)
    if point is None:
        return None
    return dict(zip(cats, point))


def verify_table(
    archive: Archive,
    source: Callable[[str], dict],
    table: Mapping[str, int],
    rounds: Sequence[Round] | None = None,
) -> dict:
    """Show, per round, that the chosen option beats every legal alternative."""
    use = list(rounds) if rounds is not None else archive.rounds
    checks = []
    ok = True
    for r in use:
        evaluated = [
            evaluate_alternative(archive, source, r, a) for a in r.alternatives
        ]
        by_id = {e["alt_id"]: e for e in evaluated}
        if r.chosen_alt is None:
            chosen_u = 0
            chosen_rec = None
        else:
            ev = by_id[r.chosen_alt]
            chosen_u = ev["allocated_qty"] * table[ev["category"]] - ev["spend_cents"]
            chosen_rec = ev
        comps = []
        for ev in evaluated:
            if not ev["signable"] or ev["alt_id"] == r.chosen_alt:
                continue
            u = ev["allocated_qty"] * table[ev["category"]] - ev["spend_cents"]
            good = chosen_u >= u
            ok = ok and good
            comps.append(
                {
                    "alt_id": ev["alt_id"],
                    "utility": u,
                    "chosen_beats": good,
                    "detail": f"{ev['allocated_qty']}*{table[ev['category']]}"
                    f" - {ev['spend_cents']} = {u}",
                }
            )
        abstain_ok = chosen_u >= 0
        ok = ok and abstain_ok
        checks.append(
            {
                "round_id": r.round_id,
                "chosen_alt": r.chosen_alt,
                "chosen_utility": chosen_u,
                "chosen_detail": (
                    None
                    if chosen_rec is None
                    else f"{chosen_rec['allocated_qty']}*"
                    f"{table[chosen_rec['category']]} - "
                    f"{chosen_rec['spend_cents']} = {chosen_u}"
                ),
                "beats_abstain": abstain_ok,
                "alternatives": comps,
                "round_ok": abstain_ok and all(c["chosen_beats"] for c in comps),
            }
        )
    return {"ok": ok, "table": dict(table), "rounds": checks}


# ---------------------------------------------------------------------------
# Infeasible round core + cheapest hand-off
# ---------------------------------------------------------------------------


def find_round_core(
    archive: Archive,
    source: Callable[[str], dict],
    rounds: Sequence[Round],
) -> list[Round]:
    """A subset of ``rounds`` that is jointly infeasible, shrunk greedily.

    Any global solution must drop at least one round of this core, which is
    what makes it a useful branch point for the hand-off search.
    """
    core = list(rounds)
    if find_valuation_table(archive, source, core) is not None:
        raise ValueError("find_round_core called on a feasible round set")
    for r in list(core):
        trial = [x for x in core if x is not r]
        if find_valuation_table(archive, source, trial) is None:
            core = trial
    return core


def _removable(r: Round) -> bool:
    return (not r.protected) and (r.review_cost is not None)


def min_cost_handoff(
    archive: Archive,
    source: Callable[[str], dict],
    round_ids: Sequence[str] | None = None,
) -> dict:
    """Cheapest set of reviewable rounds to hand off so the rest are feasible.

    Protected rounds are never removed.  Returns the hand-off set, its total
    fee, the explaining table and the conflict cores that forced the choice.
    """
    ids = set(round_ids) if round_ids is not None else {r.round_id for r in archive.rounds}
    keep_ids = ids
    rounds_by_id = {r.round_id: r for r in archive.rounds}

    def rounds_of(id_set: set[str]) -> list[Round]:
        return [rounds_by_id[i] for i in sorted(id_set)]

    best: dict = {"cost": None, "removed": None, "table": None}
    seen: dict[frozenset, int] = {}
    cores_seen: list[list[str]] = []

    def rec(kept: frozenset, removed: frozenset, cost: int) -> None:
        if best["cost"] is not None and cost >= best["cost"]:
            return
        prior = seen.get(kept)
        if prior is not None and prior <= cost:
            return
        seen[kept] = cost
        rs = rounds_of(set(kept))
        table = find_valuation_table(archive, source, rs)
        if table is not None:
            best.update(cost=cost, removed=sorted(removed), table=table)
            return
        core = find_round_core(archive, source, rs)
        core_ids = [r.round_id for r in core]
        if core_ids not in cores_seen:
            cores_seen.append(core_ids)
        removable = [r for r in core if _removable(r)]
        if not removable:
            return  # all-protected (or non-reviewable) core: cannot fix
        for r in removable:
            rec(
                kept - {r.round_id},
                removed | {r.round_id},
                cost + int(r.review_cost),
            )

    rec(frozenset(keep_ids), frozenset(), 0)
    if best["cost"] is None:
        return {
            "feasible": False,
            "reason": "no reviewable hand-off can remove the contradiction "
            "(a core is protected or not reviewable)",
            "conflict_cores": cores_seen,
        }
    return {
        "feasible": True,
        "handoff_rounds": best["removed"],
        "total_cost": best["cost"],
        "table": best["table"],
        "conflict_cores": cores_seen,
        "verification": verify_table(
            archive, source, best["table"],
            [rounds_by_id[i] for i in sorted(keep_ids - set(best["removed"]))],
        ),
    }


# ---------------------------------------------------------------------------
# Optional persistence of the auditable menu (never touches real tables)
# ---------------------------------------------------------------------------

_RESEARCH_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_menu (
    round_id   TEXT NOT NULL,
    alt_id     TEXT NOT NULL,
    payload    TEXT NOT NULL,
    PRIMARY KEY (round_id, alt_id)
);
CREATE TABLE IF NOT EXISTS research_versions (
    round_id   TEXT PRIMARY KEY,
    version    INTEGER NOT NULL,
    payload    TEXT NOT NULL
);
"""


def init_research_tables(db) -> None:
    # Avoid executescript(): it implicitly COMMITs, which breaks db.tx().
    with db.tx() as conn:
        for statement in filter(None, (s.strip() for s in _RESEARCH_SCHEMA.split(";"))):
            conn.execute(statement)


def persist_menu(db, archive: Archive, menu: Sequence[dict]) -> None:
    """Store the derived menu in research-only tables."""
    init_research_tables(db)
    with db.tx() as conn:
        conn.execute("DELETE FROM research_menu")
        for rec in menu:
            conn.execute(
                "INSERT OR REPLACE INTO research_menu(round_id, alt_id, payload) "
                "VALUES (?, ?, ?)",
                (rec["round_id"], rec["alt_id"],
                 json.dumps(rec, sort_keys=True, ensure_ascii=False)),
            )
        conn.execute("DELETE FROM research_versions")
        for r in archive.rounds:
            conn.execute(
                "INSERT INTO research_versions(round_id, version, payload) "
                "VALUES (?, ?, ?)",
                (r.round_id, r.version,
                 json.dumps(r.to_dict(), sort_keys=True, ensure_ascii=False)),
            )


def load_menu(db, round_id: str, alt_id: str) -> dict | None:
    with db.read() as conn:
        row = conn.execute(
            "SELECT payload FROM research_menu WHERE round_id=? AND alt_id=?",
            (round_id, alt_id),
        ).fetchone()
    return json.loads(row["payload"]) if row else None


def load_versions(db) -> dict[str, int]:
    with db.read() as conn:
        return {
            r["round_id"]: r["version"]
            for r in conn.execute(
                "SELECT round_id, version FROM research_versions ORDER BY round_id"
            )
        }
