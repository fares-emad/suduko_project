"""Sudoku as a constraint satisfaction problem (CSP).

Three separately switchable parts, each with its own counters:

  * Backtracking (BT)        always on; it is the base search.
  * Forward Checking (FC)    use_fc=True: after each assignment, remove that digit
                             from the domains of the 20 peers; an empty domain
                             means backtrack immediately.
  * Arc consistency (AC-3)   use_ac3=True: prune domains before the search starts.
                             mac=True additionally maintains AC-3 after every
                             assignment (the optional extension).

Cells are numbered 0..80 (row-major). A domain is an int bitmask: bit d is set when
digit d (1..9) is still possible. 0 in a grid means "blank".

Counters (CSPStats):
  assignments        number of (cell, digit) assignments tried
  backtracks         number of assignments that were undone / rejected
  fc_prunings        digits removed from domains by forward checking
  ac3_revisions      arcs processed by AC-3
  ac3_prunings       digits removed from domains by AC-3
  consistency_checks peer look-ups done by plain BT to test legality
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, fields, asdict

ALL = 0b1111111110  # digits 1..9


def _build_structure():
    rows = [[r * 9 + c for c in range(9)] for r in range(9)]
    cols = [[r * 9 + c for r in range(9)] for c in range(9)]
    boxes = [
        [(br * 3 + r) * 9 + bc * 3 + c for r in range(3) for c in range(3)]
        for br in range(3)
        for bc in range(3)
    ]
    units = rows + cols + boxes
    names = (
        [f"row {i + 1}" for i in range(9)]
        + [f"column {i + 1}" for i in range(9)]
        + [f"box {i + 1}" for i in range(9)]
    )
    peers = [set() for _ in range(81)]
    for u in units:
        for a in u:
            for b in u:
                if a != b:
                    peers[a].add(b)
    return units, names, [tuple(sorted(p)) for p in peers]


UNITS, UNIT_NAMES, PEERS = _build_structure()
ARCS = [(a, b) for a in range(81) for b in PEERS[a]]


def cell_name(i: int) -> str:
    return f"r{i // 9 + 1}c{i % 9 + 1}"


def digits_of(mask: int) -> list[int]:
    return [d for d in range(1, 10) if (mask >> d) & 1]


def find_conflicts(grid) -> list[tuple[int, int, int]]:
    """All pairs (a, b, digit) of cells that share a unit and hold the same digit."""
    seen = set()
    for u in UNITS:
        by: dict[int, list[int]] = {}
        for c in u:
            d = grid[c]
            if d:
                by.setdefault(d, []).append(c)
        for d, cells in by.items():
            if len(cells) > 1:
                for i in range(len(cells)):
                    for j in range(i + 1, len(cells)):
                        a, b = min(cells[i], cells[j]), max(cells[i], cells[j])
                        seen.add((a, b, d))
    return sorted(seen)


def explain_failure(grid) -> str:
    conf = find_conflicts(grid)
    if conf:
        a, b, d = conf[0]
        for u, name in zip(UNITS, UNIT_NAMES):
            if a in u and b in u:
                return f"digit {d} appears twice in {name} ({cell_name(a)} and {cell_name(b)})"
    for c in range(81):
        if grid[c] == 0:
            used = {grid[p] for p in PEERS[c]} - {0}
            if len(used) == 9:
                return (f"cell {cell_name(c)} has no possible digit "
                        f"(1-9 are all used in its row, column and box)")
    return "constraint propagation emptied the domain of a cell"


@dataclass
class CSPStats:
    assignments: int = 0
    backtracks: int = 0
    fc_prunings: int = 0
    ac3_revisions: int = 0
    ac3_prunings: int = 0
    consistency_checks: int = 0
    time_s: float = 0.0

    def add(self, other: "CSPStats") -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CSPResult:
    status: str  # 'solved' | 'no_solution' | 'cutoff'
    solutions: list
    n_solutions: int
    message: str
    stats: CSPStats

    @property
    def solved(self) -> bool:
        return self.status == "solved"


class Cutoff(Exception):
    pass


def ac3(dom: list[int], stats: CSPStats, queue: deque | None = None):
    """AC-3 for the binary 'not equal' constraints of Sudoku.

    Revising arc (xi, xj): if D(xj) is the single value {v}, remove v from D(xi).
    Returns (ok, wiped_cell). ok is False when some domain becomes empty.
    """
    if queue is None:
        queue = deque(ARCS)
    while queue:
        xi, xj = queue.popleft()
        stats.ac3_revisions += 1
        dj = dom[xj]
        if (dj & (dj - 1)) == 0 and (dom[xi] & dj):
            dom[xi] &= ~dj
            stats.ac3_prunings += 1
            if dom[xi] == 0:
                return False, xi
            for xk in PEERS[xi]:
                if xk != xj:
                    queue.append((xk, xi))
    return True, None


def initial_domains(grid) -> list[int]:
    return [(1 << g) if g else ALL for g in grid]


def ac3_detects_conflict(grid):
    """Run AC-3 alone on a grid. Returns dict with the verdict and domain sizes."""
    st = CSPStats()
    dom = initial_domains(grid)
    before = [bin(m).count("1") for m in dom]
    ok, wiped = ac3(dom, st)
    return {
        "conflict": not ok,
        "wiped_cell": wiped,
        "reason": None if ok else explain_failure(grid),
        "domain_sizes_before": before,
        "domain_sizes_after": [bin(m).count("1") for m in dom],
        "stats": st,
    }


class SudokuCSP:
    def __init__(self, grid, use_fc=False, use_ac3=False, mac=False,
                 ordering="mrv", max_assignments=None, rng=None):
        if len(grid) != 81:
            raise ValueError("grid must have 81 entries")
        self.grid = [int(x) for x in grid]
        self.use_fc = bool(use_fc or mac)
        self.use_ac3 = bool(use_ac3 or mac)
        self.mac = bool(mac)
        if ordering not in ("mrv", "static"):
            raise ValueError("ordering must be 'mrv' or 'static'")
        self.ordering = ordering
        self.max_assignments = max_assignments
        self.rng = rng
        self.stats = CSPStats()

    # ------------------------------------------------------------------ helpers
    def _fail(self, msg: str) -> CSPResult:
        return CSPResult("no_solution", [], 0, "NO SOLUTION: " + msg, self.stats)

    def _legal_nofc(self, dom, val, c):
        m = dom[c]
        for p in PEERS[c]:
            pv = val[p]
            if pv:
                m &= ~(1 << pv)
        self.stats.consistency_checks += 20
        return m

    def _select(self, dom, val):
        best, bestm, bestn = -1, 0, 10
        for c in range(81):
            if val[c]:
                continue
            m = dom[c] if self.use_fc else self._legal_nofc(dom, val, c)
            if self.ordering == "static":
                return c, m
            n = m.bit_count()
            if n < bestn:
                best, bestm, bestn = c, m, n
                if n <= 1:
                    break
        return best, bestm

    # ------------------------------------------------------------------- search
    def _rec(self, dom, val):
        st = self.stats
        c, m = self._select(dom, val)
        if c == -1:
            self._sols.append(val[:])
            return
        values = digits_of(m)
        if self.rng is not None:
            self.rng.shuffle(values)
        for v in values:
            st.assignments += 1
            if self.max_assignments and st.assignments > self.max_assignments:
                raise Cutoff
            if self.use_fc:
                nd = dom[:]
                bit = 1 << v
                nd[c] = bit
                ok = True
                changed = []
                for p in PEERS[c]:
                    if nd[p] & bit:
                        nd[p] &= ~bit
                        st.fc_prunings += 1
                        changed.append(p)
                        if nd[p] == 0:
                            ok = False
                            break
                if ok and self.mac and changed:
                    q = deque((xk, p) for p in changed for xk in PEERS[p])
                    ok, _ = ac3(nd, st, q)
                if not ok:
                    st.backtracks += 1
                    continue
                val[c] = v
                self._rec(nd, val)
            else:
                val[c] = v
                self._rec(dom, val)
            val[c] = 0
            if len(self._sols) >= self._max:
                return
            st.backtracks += 1

    # -------------------------------------------------------------------- solve
    def solve(self, max_solutions: int = 1) -> CSPResult:
        t0 = time.perf_counter()
        try:
            return self._solve(max_solutions)
        finally:
            self.stats.time_s += time.perf_counter() - t0

    def _solve(self, max_solutions):
        grid, st = self.grid, self.stats
        self._max = max_solutions
        self._sols: list = []
        dom = initial_domains(grid)

        if not (self.use_fc or self.use_ac3):
            st.consistency_checks += 27
            if find_conflicts(grid):
                return self._fail(explain_failure(grid))
        if self.use_ac3:
            ok, wiped = ac3(dom, st)
            if not ok:
                return self._fail(f"AC-3 emptied the domain of {cell_name(wiped)}: "
                                  + explain_failure(grid))
        if self.use_fc:
            for c in range(81):
                if grid[c]:
                    bit = 1 << grid[c]
                    for p in PEERS[c]:
                        if dom[p] & bit:
                            dom[p] &= ~bit
                            st.fc_prunings += 1
                            if dom[p] == 0:
                                return self._fail("forward checking emptied the domain of "
                                                  f"{cell_name(p)}: " + explain_failure(grid))
        val = list(grid)
        try:
            self._rec(dom, val)
        except Cutoff:
            return CSPResult("cutoff", self._sols, len(self._sols),
                             f"CUTOFF after {st.assignments - 1} assignments", st)
        if self._sols:
            return CSPResult("solved", self._sols, len(self._sols), "solved", st)
        return self._fail(f"search space exhausted after {st.backtracks} backtracks")


def count_solutions(grid, limit=2) -> int:
    r = SudokuCSP(grid, use_fc=True, use_ac3=True).solve(max_solutions=limit)
    return r.n_solutions


def is_valid_solution(solution, puzzle=None) -> bool:
    if len(solution) != 81 or any(not (1 <= d <= 9) for d in solution):
        return False
    if find_conflicts(solution):
        return False
    if puzzle is not None and any(p and p != s for p, s in zip(puzzle, solution)):
        return False
    return True
