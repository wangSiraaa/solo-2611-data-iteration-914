"""Tiny exact integer-linear feasibility solver (pure Python).

The research layer asks a single question repeatedly: *is there an integer
valuation vector satisfying these linear inequalities, and if so which one?*
No SAT/LP package is available in this deployment, so we implement it here:

* two-phase simplex for the **LP relaxation** (Phase I only -- we never need
  to optimise an objective, just find a feasible real point), using exact
  :class:`fractions.Fraction` arithmetic so no floating-point lies sneak in;
* branch-and-bound over integer variables, branching on a fractional value
  from the LP solution.

A returned witness is therefore a genuine integer point satisfying every
constraint, not a rounded float approximation.

Constraints use the form ``sum(coeff * x) (<= | >= | ==) rhs`` with
``x >= 0``.  Integer lower/upper bounds are supplied separately.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Sequence

# Constraint wire format: (coeffs: Sequence[int], sense: str, rhs: int)
Constraint = tuple[Sequence[int], str, int]

# Guard against pathological input in synthetic material.
MAX_NODES = 400_000


class _Infeasible(Exception):
    pass


def _normalise(n: int, rows: Iterable[Constraint]):
    """Return (coeffs, sense, rhs) with rhs >= 0, coercing to Fraction."""
    out = []
    flip = {"<=": ">=", ">=": "<=", "==": "=="}
    for coeffs, sense, rhs in rows:
        if sense not in flip:
            raise ValueError(f"bad sense {sense!r}")
        a = [Fraction(c) for c in coeffs]
        if len(a) != n:
            raise ValueError("constraint width mismatch")
        r = Fraction(rhs)
        s = sense
        if r < 0:
            a = [-x for x in a]
            r = -r
            s = flip[s]
        if all(x == 0 for x in a):
            # Here rhs >= 0.  0 <= r is always true; 0 >= r and 0 == r hold
            # only when r == 0.
            if s != "<=" and r > 0:
                raise _Infeasible("0 op rhs with rhs > 0")
            continue
        out.append((a, s, r))
    return out


def _lp_feasible(n: int, rows: Sequence[Constraint]) -> list[Fraction] | None:
    """Phase-I simplex: find x >= 0 with all rows satisfied, or None."""
    try:
        norm = _normalise(n, rows)
    except _Infeasible:
        return None
    m = len(norm)
    if m == 0:
        return [Fraction(0)] * n

    # Allocate auxiliary columns.
    n_total = n
    extras: list[dict[str, int]] = []
    for _, sense, _ in norm:
        cols: dict[str, int] = {}
        if sense == "<=":
            cols["slack"] = n_total
            n_total += 1
        elif sense == ">=":
            cols["surplus"] = n_total
            n_total += 1
            cols["art"] = n_total
            n_total += 1
        else:  # ==
            cols["art"] = n_total
            n_total += 1
        extras.append(cols)

    tab: list[list[Fraction]] = []
    basis: list[int] = []
    art_cols: list[int] = []
    for i, (a, sense, r) in enumerate(norm):
        row = [Fraction(0)] * (n_total + 1)
        for j in range(n):
            row[j] = a[j]
        row[n_total] = r
        cols = extras[i]
        if sense == "<=":
            row[cols["slack"]] = Fraction(1)
            basis.append(cols["slack"])
        elif sense == ">=":
            row[cols["surplus"]] = Fraction(-1)
            row[cols["art"]] = Fraction(1)
            basis.append(cols["art"])
            art_cols.append(cols["art"])
        else:
            row[cols["art"]] = Fraction(1)
            basis.append(cols["art"])
            art_cols.append(cols["art"])
        tab.append(row)

    # Phase-I objective: maximise -sum(artificial).  Reduced-cost row.
    c = [Fraction(0)] * n_total
    for j in art_cols:
        c[j] = Fraction(-1)
    obj = c[:] + [Fraction(0)]
    for i in range(m):
        cb = c[basis[i]]
        if cb != 0:
            row = tab[i]
            for j in range(n_total + 1):
                obj[j] -= cb * row[j]

    # Bland's rule guarantees termination on degenerate tableaux.
    while True:
        enter = -1
        for j in range(n_total):
            if obj[j] > 0:
                enter = j
                break
        if enter == -1:
            break
        leave = -1
        best_ratio: Fraction | None = None
        for i in range(m):
            a = tab[i][enter]
            if a > 0:
                ratio = tab[i][n_total] / a
                if (
                    best_ratio is None
                    or ratio < best_ratio
                    or (ratio == best_ratio and basis[i] < basis[leave])
                ):
                    best_ratio = ratio
                    leave = i
        if leave == -1:
            # Phase-I objective is bounded above by 0, so this cannot happen
            # for well-formed input; treat as infeasible defensively.
            return None
        piv = tab[leave][enter]
        if piv != 1:
            tab[leave] = [x / piv for x in tab[leave]]
        for i in range(m):
            if i != leave:
                f = tab[i][enter]
                if f != 0:
                    row_l = tab[leave]
                    tab[i] = [tab[i][j] - f * row_l[j] for j in range(n_total + 1)]
        f = obj[enter]
        if f != 0:
            row_l = tab[leave]
            obj = [obj[j] - f * row_l[j] for j in range(n_total + 1)]
        basis[leave] = enter

    # Feasible iff no artificial remains basic at a positive value.
    art_set = set(art_cols)
    for i in range(m):
        if basis[i] in art_set and tab[i][n_total] > 0:
            return None

    x = [Fraction(0)] * n
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = tab[i][n_total]
    return x


def integer_feasible(
    n: int,
    rows: Sequence[Constraint],
    lo: Sequence[int] | None = None,
    hi: Sequence[int] | None = None,
) -> list[int] | None:
    """Return an integer x with lo <= x <= hi satisfying ``rows``, or None.

    ``rows`` use variables x >= 0 (which is also the implicit lower bound).
    """
    lo = list(lo) if lo is not None else [0] * n
    hi = list(hi) if hi is not None else [10 ** 9] * n
    for j in range(n):
        if lo[j] < 0:
            raise ValueError("integer_feasible assumes lo >= 0")
        if lo[j] > hi[j]:
            return None
    if n == 0:
        return [] if _lp_feasible(0, rows) is not None else None

    nodes = [0]

    def solve(cur_lo: list[int], cur_hi: list[int]):
        nodes[0] += 1
        if nodes[0] > MAX_NODES:
            raise RuntimeError("integer feasibility search exceeded node budget")
        all_rows = list(rows)
        for j in range(n):
            e = [0] * n
            e[j] = 1
            all_rows.append((e, "<=", cur_hi[j]))
            if cur_lo[j] > 0:
                e2 = [0] * n
                e2[j] = -1
                all_rows.append((e2, "<=", -cur_lo[j]))
        return _lp_feasible(n, all_rows)

    def rec(cur_lo: list[int], cur_hi: list[int]):
        x = solve(cur_lo, cur_hi)
        if x is None:
            return None
        frac_j = -1
        for j in range(n):
            if x[j].denominator != 1:
                frac_j = j
                break
        if frac_j == -1:
            return [int(v) for v in x]
        j = frac_j
        floor_j = x[j].numerator // x[j].denominator  # x[j] >= 0
        hi2 = cur_hi[:]
        hi2[j] = min(hi2[j], floor_j)
        if cur_lo[j] <= hi2[j]:
            got = rec(cur_lo[:], hi2)
            if got is not None:
                return got
        lo2 = cur_lo[:]
        lo2[j] = max(lo2[j], floor_j + 1)
        if lo2[j] <= cur_hi[j]:
            got = rec(lo2, cur_hi[:])
            if got is not None:
                return got
        return None

    return rec(lo, hi)


def any_integer_point(
    n: int, rows: Sequence[Constraint], hi: Sequence[int]
) -> list[int] | None:
    return integer_feasible(n, rows, [0] * n, hi)
