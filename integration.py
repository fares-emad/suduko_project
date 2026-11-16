"""The integrated pipeline: NN reads -> AC-3 finds conflicts -> Search corrects -> CSP solves.

  1. Blank cells are found by a brightness rule (mean pixel < BLANK_THRESHOLD), not by the NN.
  2. The NN reads every non-blank cell and gives calibrated probabilities.
  3. AC-3 runs on the first reading; an emptied domain proves a conflict.
  4. Search (UCS / Greedy / A*) proposes sets of corrections, cheapest first; the CSP
     tests each corrected grid. The first solvable one wins.
  5. If no set of at most `max_corrections` works, the result is "UNREADABLE" and the
     least confident cells are listed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .csp import ac3_detects_conflict, cell_name
from .search import CorrectionProblem, SearchResult, check_grid, run_search

BLANK_THRESHOLD = 0.03


def read_mosaic(clf, images: np.ndarray, blank_threshold: float = BLANK_THRESHOLD):
    """Return (read_grid, probs (81, 9)). Blank cells -> 0 and a zero probability row."""
    blank = images.mean(axis=1) < blank_threshold
    probs = np.zeros((81, 9))
    grid = [0] * 81
    if (~blank).any():
        P = clf.predict_proba(images[~blank])
        probs[~blank] = P
        idx = np.where(~blank)[0]
        for c, p in zip(idx, P):
            grid[int(c)] = int(p.argmax()) + 1
    return grid, probs


@dataclass
class PipelineResult:
    status: str  # 'solved' | 'unreadable'
    message: str
    read_grid: list
    probs: np.ndarray
    ac3_info: dict
    search: SearchResult | None
    corrections: list = field(default_factory=list)
    solution: list | None = None
    least_confident: list = field(default_factory=list)

    def correction_log(self):
        return [{"cell": cell_name(c.cell), "read_digit": c.read, "new_digit": c.new,
                 "cost": round(c.cost, 4)} for c in self.corrections]


def least_confident_cells(grid, probs, k=5):
    given = [c for c in range(81) if grid[c]]
    order = sorted(given, key=lambda c: probs[c].max())[:k]
    return [(cell_name(c), grid[c], float(probs[c].max())) for c in order]


def solve_mosaic(clf, images, algo="ucs", max_corrections=3, n_uncertain=12, n_alts=3,
                 require_unique=False, node_limit=50000, trace_limit=40) -> PipelineResult:
    grid, probs = read_mosaic(clf, images)
    return solve_reading(grid, probs, algo, max_corrections, n_uncertain, n_alts,
                         require_unique, node_limit, trace_limit)


def solve_reading(grid, probs, algo="ucs", max_corrections=3, n_uncertain=12, n_alts=3,
                  require_unique=False, node_limit=50000, trace_limit=40) -> PipelineResult:
    info = ac3_detects_conflict(grid)
    prob = CorrectionProblem(grid, probs, max_corrections, n_uncertain, n_alts, require_unique)
    res = run_search(prob, algo, node_limit=node_limit, trace_limit=trace_limit)
    if res.found:
        n = len(res.corrections)
        msg = ("solved with no corrections" if n == 0
               else f"solved after {n} correction(s), total cost {res.total_cost:.3f}")
        return PipelineResult("solved", msg, grid, probs, info, res, res.corrections, res.solution)
    lc = least_confident_cells(grid, probs)
    why = "search limit reached" if res.status == "limit" else "no correction set found"
    msg = (f"UNREADABLE: {why} with at most {max_corrections} corrections. "
           "Least confident cells: "
           + ", ".join(f"{n} read as {d} (p={p:.2f})" for n, d, p in lc))
    return PipelineResult("unreadable", msg, grid, probs, info, res, [], None, lc)


# ---------------------------------------------------------------- baselines (exp. 3)
def solve_without_correction(grid, require_unique=False):
    """'NN only': accept the first reading if it is solvable, otherwise give up."""
    ok, sol, _ = check_grid(grid, require_unique)
    return ok, sol


def solve_random_correction(grid, rng: np.random.Generator, budget=500, max_corrections=3,
                            require_unique=False):
    """Random baseline: try random correction sets (1..max cells, random new digits)."""
    ok, sol, _ = check_grid(grid, require_unique)
    if ok:
        return True, sol, 0
    given = [c for c in range(81) if grid[c]]
    for t in range(1, budget + 1):
        k = int(rng.integers(1, max_corrections + 1))
        cells = rng.choice(given, size=k, replace=False)
        g = list(grid)
        for c in cells:
            g[int(c)] = int(rng.choice([d for d in range(1, 10) if d != grid[int(c)]]))
        ok, sol, _ = check_grid(g, require_unique)
        if ok:
            return True, sol, t
    return False, None, budget
