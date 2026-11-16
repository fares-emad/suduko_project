"""CSP tests: 5+ normal, 3 edge, 2 impossible cases (spread over the test files)."""
import itertools

import pytest

from conftest import PUZZLE, SOLUTION
from sudoku_ai.csp import (SudokuCSP, ac3_detects_conflict, count_solutions,
                           find_conflicts, is_valid_solution)
from sudoku_ai.gen import generate_puzzle

CONFIGS = [dict(), dict(use_fc=True), dict(use_fc=True, use_ac3=True),
           dict(mac=True)]
IDS = ["BT", "BT+FC", "BT+FC+AC3", "MAC"]


# ----------------------------------------------------------------- normal cases
@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_normal_known_puzzle_every_configuration(cfg):  # normal 1-4
    r = SudokuCSP(PUZZLE, **cfg).solve()
    assert r.solved and r.solutions[0] == SOLUTION
    assert r.stats.assignments > 0


def test_normal_generated_puzzle_unique_and_solved():  # normal 5
    puz, sol = generate_puzzle(30, seed=3)
    assert sum(1 for x in puz if x) == 30
    assert count_solutions(puz, limit=2) == 1
    r = SudokuCSP(puz, use_fc=True, use_ac3=True).solve()
    assert r.solutions[0] == sol and is_valid_solution(r.solutions[0], puz)


def test_normal_forward_checking_prunes_and_counts():  # normal 6
    r = SudokuCSP(PUZZLE, use_fc=True).solve()
    assert r.stats.fc_prunings > 0
    assert r.stats.consistency_checks == 0  # FC does not use plain BT checks
    r2 = SudokuCSP(PUZZLE).solve()
    assert r2.stats.fc_prunings == 0 and r2.stats.consistency_checks > 0


def test_normal_ac3_prunes_domains_before_search():  # normal 7
    info = ac3_detects_conflict(PUZZLE)
    assert not info["conflict"]
    assert sum(info["domain_sizes_after"]) < sum(info["domain_sizes_before"])
    assert info["stats"].ac3_prunings > 0


# ------------------------------------------------------------------- edge cases
@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_edge_already_complete_grid(cfg):  # edge 1
    r = SudokuCSP(SOLUTION, **cfg).solve()
    assert r.solved and r.solutions[0] == SOLUTION and r.stats.assignments == 0


def test_edge_empty_grid_is_solvable():  # edge 2
    r = SudokuCSP([0] * 81, use_fc=True, use_ac3=True).solve()
    assert r.solved and is_valid_solution(r.solutions[0])
    assert count_solutions([0] * 81, limit=2) == 2


def test_edge_single_blank_cell_filled_correctly():  # edge 3
    g = list(SOLUTION)
    g[40] = 0
    r = SudokuCSP(g, use_fc=True).solve()
    assert r.solutions[0][40] == SOLUTION[40] and r.n_solutions == 1


def test_edge_cutoff_is_reported_not_hidden():  # edge 4
    r = SudokuCSP([0] * 81, ordering="static", max_assignments=5).solve()
    assert r.status == "cutoff" and "CUTOFF" in r.message


# --------------------------------------------------------------- impossible cases
@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_impossible_duplicate_givens_clear_message(cfg):  # impossible 1
    g = list(PUZZLE)
    g[1] = 5  # r1c1 is already 5
    assert find_conflicts(g)
    r = SudokuCSP(g, **cfg).solve()
    assert r.status == "no_solution"
    assert r.message.startswith("NO SOLUTION") and "row 1" in r.message


def test_impossible_ac3_alone_detects_the_conflict():  # impossible 1b
    g = list(PUZZLE)
    g[1] = 5
    info = ac3_detects_conflict(g)
    assert info["conflict"] and info["wiped_cell"] is not None


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_impossible_cell_with_no_legal_digit(cfg):  # impossible 2
    g = [0] * 81
    g[1:8] = [2, 3, 4, 5, 6, 7, 8]  # r1c2..r1c8 (row 1)
    g[9], g[10] = 1, 9              # r2c1 (column 1), r2c2 (box 1)
    assert not find_conflicts(g)
    r = SudokuCSP(g, **cfg).solve()
    assert r.status == "no_solution" and "NO SOLUTION" in r.message


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_impossible_last_cell_of_row_blocked_by_column(cfg):  # impossible 3
    g = [0] * 81
    g[0:8] = [1, 2, 3, 4, 5, 6, 7, 8]  # row 1 needs a 9 in r1c9 ...
    g[17] = 9                          # ... but r2c9 (same column) is 9
    assert not find_conflicts(g)
    r = SudokuCSP(g, **cfg).solve()
    assert r.status == "no_solution" and r.message.startswith("NO SOLUTION")
