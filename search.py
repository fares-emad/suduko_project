"""Search over sets of corrections to the network's first reading of the grid.

State    : a tuple of (cell, new_digit) pairs, sorted by cell; at most one per cell.
Start    : () = the network's raw reading.
Action   : change one more uncertain cell to one of its top alternative digits.
Step cost: max(0, log p(read digit) - log p(new digit))  (never negative, 0 only
           for exact ties). Cost of a state g = sum of its step costs, so UCS
           returns the most likely correction set.
Goal     : the corrected grid has a solution (optionally: exactly one solution),
           decided by the CSP (FC + AC-3). A cheap duplicate-digit scan rejects
           grids that obviously break a row/column/box rule before the CSP runs.
Limit    : at most `max_corrections` corrections (default 3).

Algorithms (all with counters):
  ucs     priority g                          -> optimal
  greedy  priority h                          -> fast, not optimal
  astar   priority g + h, h admissible        -> optimal (states may be re-opened)

Heuristic h ("each duplicate-digit conflict needs at least one changed cell"):
  take the conflicting pairs of the current corrected grid, greedily pick pairs
  that share no cell, and for each picked pair add the cheapest possible
  correction cost of its two cells (cells already corrected or without
  alternatives cost infinity). Every picked pair needs its own correction, so h
  never overestimates the remaining cost. If the picked pairs outnumber the
  remaining corrections, or some pair cannot be fixed, h = infinity (dead state).
"""
from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .csp import CSPStats, SudokuCSP, find_conflicts

INF = float("inf")
EPS = 1e-12


@dataclass(frozen=True)
class Correction:
    cell: int
    read: int
    new: int
    cost: float


@dataclass
class SearchResult:
    algorithm: str
    status: str  # 'found' | 'exhausted' | 'limit'
    state: tuple | None
    corrections: list
    total_cost: float
    nodes_generated: int
    nodes_expanded: int
    goal_tests: int
    quick_rejects: int
    csp_calls: int
    csp_stats: CSPStats
    solution: list | None
    time_s: float
    trace: list = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.status == "found"


def check_grid(grid, unique: bool = False, stats: CSPStats | None = None):
    """Goal test. Returns (ok, solution, used_csp)."""
    if find_conflicts(grid):
        return False, None, False
    csp = SudokuCSP(grid, use_fc=True, use_ac3=True)
    r = csp.solve(max_solutions=2 if unique else 1)
    if stats is not None:
        stats.add(csp.stats)
    ok = r.solved and (not unique or r.n_solutions == 1)
    return ok, (r.solutions[0] if ok else None), True


class CorrectionProblem:
    def __init__(self, read_grid, probs, max_corrections=3, n_uncertain=12,
                 n_alts=3, require_unique=False):
        self.grid = list(read_grid)
        self.probs = np.asarray(probs, dtype=float)
        self.max_corrections = max_corrections
        self.require_unique = require_unique
        given = [c for c in range(81) if self.grid[c]]
        order = sorted(given, key=lambda c: (self.probs[c].max(), c))
        self.cells = sorted(order[:n_uncertain])
        self.cand: dict[int, list[tuple[int, float]]] = {}
        for c in self.cells:
            p = np.clip(self.probs[c], 1e-12, 1.0)
            read = self.grid[c]
            alts = [d for d in np.argsort(-p) + 1 if d != read][:n_alts]
            lp_read = math.log(p[read - 1])
            self.cand[c] = [(int(d), max(0.0, lp_read - math.log(p[d - 1]))) for d in alts]
        self.cmin = {c: min(cost for _, cost in v) for c, v in self.cand.items() if v}
        # counters
        self.goal_tests = 0
        self.quick_rejects = 0
        self.csp_calls = 0
        self.csp_stats = CSPStats()
        self._goal_cache: dict = {}
        self._solutions: dict = {}
        self._h_cache: dict = {}

    def apply(self, state):
        g = list(self.grid)
        for c, d in state:
            g[c] = d
        return g

    def goal(self, state) -> bool:
        if state in self._goal_cache:
            return self._goal_cache[state]
        self.goal_tests += 1
        ok, sol, used = check_grid(self.apply(state), self.require_unique, self.csp_stats)
        if used:
            self.csp_calls += 1
        else:
            self.quick_rejects += 1
        self._goal_cache[state] = ok
        if ok:
            self._solutions[state] = sol
        return ok

    def solution(self, state):
        return self._solutions.get(state)

    def h(self, state) -> float:
        if state in self._h_cache:
            return self._h_cache[state]
        conf = find_conflicts(self.apply(state))
        if not conf:
            val = 0.0
        else:
            used = {c for c, _ in state}

            def cm(c):
                return INF if c in used else self.cmin.get(c, INF)

            pairs = sorted(((min(cm(a), cm(b)), a, b) for a, b, _ in conf), reverse=True)
            taken, total, npairs = set(), 0.0, 0
            val = None
            for m, a, b in pairs:
                if a in taken or b in taken:
                    continue
                if m == INF:
                    val = INF
                    break
                taken.update((a, b))
                total += m
                npairs += 1
            if val is None:
                val = INF if npairs > self.max_corrections - len(state) else total
        self._h_cache[state] = val
        return val

    def neighbors(self, state):
        if len(state) >= self.max_corrections:
            return []
        used = {c for c, _ in state}
        out = []
        for c in self.cells:
            if c in used:
                continue
            for d, cost in self.cand[c]:
                out.append((tuple(sorted(state + ((c, d),))), cost))
        return out

    def corrections_of(self, state):
        res = []
        for c, d in state:
            cost = dict(self.cand[c])[d]
            res.append(Correction(c, self.grid[c], d, cost))
        return res


def run_search(problem: CorrectionProblem, algo: str = "ucs", node_limit: int = 50000,
               trace_limit: int = 40) -> SearchResult:
    if algo not in ("ucs", "greedy", "astar"):
        raise ValueError("algo must be ucs, greedy or astar")
    t0 = time.perf_counter()
    counter = itertools.count()
    start: tuple = ()
    h0 = 0.0 if algo == "ucs" else problem.h(start)
    best_g = {start: 0.0}
    prio0 = {"ucs": 0.0, "greedy": h0, "astar": h0}[algo]
    heap = [(prio0, next(counter), 0.0, start)]
    generated, expanded = 1, 0
    trace: list = []
    status, found_state = "exhausted", None
    if h0 == INF:
        heap = []
    while heap:
        _, _, g, state = heapq.heappop(heap)
        if g > best_g.get(state, INF) + EPS:
            continue  # stale entry
        expanded += 1
        if len(trace) < trace_limit:
            h = None if algo == "ucs" else problem.h(state)
            trace.append({"n": expanded, "state": state, "g": g, "h": h})
        if problem.goal(state):
            status, found_state = "found", state
            break
        if expanded >= node_limit:
            status = "limit"
            break
        for ns, cost in problem.neighbors(state):
            ng = g + cost
            if ng >= best_g.get(ns, INF) - EPS:
                continue
            if algo == "ucs":
                prio = ng
            else:
                h = problem.h(ns)
                if h == INF:
                    continue  # dead state
                prio = h if algo == "greedy" else ng + h
            best_g[ns] = ng
            generated += 1
            heapq.heappush(heap, (prio, next(counter), ng, ns))
    corr = problem.corrections_of(found_state) if found_state is not None else []
    return SearchResult(
        algorithm=algo, status=status, state=found_state, corrections=corr,
        total_cost=sum(c.cost for c in corr) if found_state is not None else INF,
        nodes_generated=generated, nodes_expanded=expanded,
        goal_tests=problem.goal_tests, quick_rejects=problem.quick_rejects,
        csp_calls=problem.csp_calls, csp_stats=problem.csp_stats,
        solution=problem.solution(found_state) if found_state is not None else None,
        time_s=time.perf_counter() - t0, trace=trace,
    )
