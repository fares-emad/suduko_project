"""Search tests: known answer, brute-force check, admissibility, optimality, impossible."""
import itertools
import math

import numpy as np
import pytest

from conftest import PUZZLE, SOLUTION, synthetic_probs
from sudoku_ai.csp import find_conflicts
from sudoku_ai.integration import solve_reading
from sudoku_ai.search import INF, CorrectionProblem, run_search


def misread(grid, cell, new):
    g = list(grid)
    g[cell] = new
    return g


def brute_force_min_cost(problem):
    """Enumerate every correction set (<= max_corrections) over the candidate pool."""
    best, best_set = INF, None
    options = [(c, d, cost) for c in problem.cells for d, cost in problem.cand[c]]
    for k in range(0, problem.max_corrections + 1):
        for combo in itertools.combinations(options, k):
            cells = [c for c, _, _ in combo]
            if len(set(cells)) < k:
                continue
            state = tuple(sorted((c, d) for c, d, _ in combo))
            if problem.goal(state):
                cost = sum(x for _, _, x in combo)
                if cost < best - 1e-12:
                    best, best_set = cost, state
    return best, best_set


def make_problem(misreads, n_unc=6, n_alts=2, max_corr=3, unique=False):
    grid = list(PUZZLE)
    low = {}
    for cell, true_digit in misreads.items():
        grid[cell] = (true_digit % 9) + 1  # the network's wrong reading
    for cell, true_digit in misreads.items():
        low[cell] = true_digit
    # the givens must still carry the true digit for the cell to be a given
    probs = synthetic_probs(grid, low)
    return grid, CorrectionProblem(grid, probs, max_corr, n_unc, n_alts, unique)


GIVEN_CELLS = [c for c, d in enumerate(PUZZLE) if d]


def test_normal_ucs_finds_the_known_correction():  # known answer
    cell = GIVEN_CELLS[3]
    grid, prob = make_problem({cell: PUZZLE[cell]})
    res = run_search(prob, "ucs")
    assert res.found and len(res.corrections) == 1
    c = res.corrections[0]
    assert (c.cell, c.new) == (cell, PUZZLE[cell])
    assert math.isclose(c.cost, math.log(0.50 / 0.45), abs_tol=1e-9)
    assert res.solution == SOLUTION


@pytest.mark.parametrize("cells", [(3,), (3, 9), (0, 12, 25)], ids=["1 misread", "2", "3"])
def test_normal_ucs_matches_brute_force(cells):
    mis = {GIVEN_CELLS[i]: PUZZLE[GIVEN_CELLS[i]] for i in cells}
    grid, prob = make_problem(mis, n_unc=6, n_alts=2)
    res = run_search(prob, "ucs")
    bf_cost, bf_set = brute_force_min_cost(prob)
    assert res.found
    assert math.isclose(res.total_cost, bf_cost, abs_tol=1e-9)


@pytest.mark.parametrize("unique", [False, True])
def test_normal_astar_and_ucs_have_equal_optimal_cost(unique):
    mis = {GIVEN_CELLS[i]: PUZZLE[GIVEN_CELLS[i]] for i in (2, 7)}
    grid, p1 = make_problem(mis, unique=unique)
    _, p2 = make_problem(mis, unique=unique)
    _, p3 = make_problem(mis, unique=unique)
    u, a, g = run_search(p1, "ucs"), run_search(p2, "astar"), run_search(p3, "greedy")
    assert u.found and a.found and g.found
    assert math.isclose(u.total_cost, a.total_cost, abs_tol=1e-9)
    assert g.total_cost >= u.total_cost - 1e-9  # greedy is never better than optimal
    assert a.nodes_expanded <= u.nodes_expanded


def test_normal_heuristic_never_overestimates():
    mis = {GIVEN_CELLS[i]: PUZZLE[GIVEN_CELLS[i]] for i in (1, 5, 8)}
    grid, prob = make_problem(mis, n_unc=6, n_alts=2)
    options = [(c, d, cost) for c in prob.cells for d, cost in prob.cand[c]]
    checked = 0
    for k in range(0, 3):
        for combo in itertools.combinations(options, k):
            if len({c for c, _, _ in combo}) < k:
                continue
            state = tuple(sorted((c, d) for c, d, _ in combo))
            h = prob.h(state)
            # true remaining cost: cheapest extension of `state` to a goal
            best = INF
            rest = [o for o in options if o[0] not in {c for c, _, _ in combo}]
            for extra in range(0, prob.max_corrections - k + 1):
                for ext in itertools.combinations(rest, extra):
                    if len({c for c, _, _ in ext}) < extra:
                        continue
                    s2 = tuple(sorted(state + tuple((c, d) for c, d, _ in ext)))
                    if prob.goal(s2):
                        best = min(best, sum(x for _, _, x in ext))
            assert h <= best + 1e-9, (state, h, best)
            checked += 1
    assert checked > 50


def test_normal_counters_are_filled():
    mis = {GIVEN_CELLS[4]: PUZZLE[GIVEN_CELLS[4]]}
    _, prob = make_problem(mis)
    res = run_search(prob, "ucs")
    assert res.nodes_expanded >= 1 and res.nodes_generated >= res.nodes_expanded
    assert res.goal_tests >= 1 and res.csp_stats.assignments >= 0


# ------------------------------------------------------------------- edge cases
def test_edge_already_readable_needs_zero_corrections():
    probs = synthetic_probs(PUZZLE, {})
    res = solve_reading(list(PUZZLE), probs)
    assert res.status == "solved" and res.corrections == [] and res.solution == SOLUTION


def test_edge_max_corrections_zero_only_tests_the_reading():
    probs = synthetic_probs(PUZZLE, {})
    res = solve_reading(list(PUZZLE), probs, max_corrections=0)
    assert res.status == "solved"


def test_edge_correction_costs_are_never_negative():
    cell = GIVEN_CELLS[10]
    grid, prob = make_problem({cell: PUZZLE[cell]})
    for c in prob.cells:
        assert all(cost >= 0 for _, cost in prob.cand[c])


# --------------------------------------------------------------- impossible cases
def test_impossible_too_many_misreads_gives_unreadable_report():
    cells = [GIVEN_CELLS[i] for i in (0, 4, 8, 12)]
    grid = list(PUZZLE)
    low = {}
    for c in cells:
        grid[c] = (PUZZLE[c] % 9) + 1
        low[c] = PUZZLE[c]
    probs = synthetic_probs(grid, low)
    res = solve_reading(grid, probs, max_corrections=3, n_uncertain=8)
    assert res.status == "unreadable"
    assert res.message.startswith("UNREADABLE") and "Least confident cells" in res.message
    assert len(res.least_confident) == 5 and res.solution is None


def test_impossible_conflict_but_no_corrections_allowed():
    g = list(PUZZLE)
    g[1] = 5  # duplicate 5 in row 1
    probs = synthetic_probs(g, {})
    res = solve_reading(g, probs, max_corrections=0)
    assert res.status == "unreadable" and res.ac3_info["conflict"]
    assert "UNREADABLE" in res.message
