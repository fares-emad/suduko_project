"""Network, generator and integration tests (use the real digit data)."""
import numpy as np

from conftest import PUZZLE, SOLUTION
from sudoku_ai.gen import HardImageBank, build_mosaic, generate_puzzle
from sudoku_ai.integration import read_mosaic, solve_mosaic
from sudoku_ai.nn import expected_calibration_error, nll


def test_network_accuracy_and_probabilities(clf, digits):
    data, _ = digits
    P = clf.predict_proba(data["Xte"])
    assert P.shape == (len(data["yte"]), 9)
    assert np.allclose(P.sum(axis=1), 1.0)
    acc = (P.argmax(1) + 1 == data["yte"]).mean()
    assert acc > 0.90
    assert 0.0 <= expected_calibration_error(P, data["yte"] - 1) <= 0.2
    assert nll(P, data["yte"] - 1) < 0.5


def test_generator_is_seeded():
    a = generate_puzzle(32, seed=11)
    b = generate_puzzle(32, seed=11)
    assert a == b


def test_blank_cells_detected_by_brightness_rule(clf, digits):
    data, _ = digits
    rng = np.random.default_rng(0)
    imgs, _ = build_mosaic(PUZZLE, data["Xte"], data["yte"], rng)
    grid, probs = read_mosaic(clf, imgs)
    assert [c for c in range(81) if grid[c] == 0] == [c for c in range(81) if PUZZLE[c] == 0]
    assert np.allclose(probs[[c for c in range(81) if PUZZLE[c] == 0]], 0)


def test_hard_digits_really_are_misread_but_plausible(clf, digits):
    data, _ = digits
    bank = HardImageBank(clf, data["Xte"], data["yte"], n_per_digit=8, seed=1)
    for d in range(1, 10):
        for x in bank.images[d]:
            p = clf.predict_proba(x[None, :])[0]
            assert p.argmax() + 1 != d
            assert d in (np.argsort(-p)[:3] + 1)


def test_end_to_end_recovers_the_true_solution(clf, digits):
    data, _ = digits
    puz, sol = generate_puzzle(32, seed=21)
    bank = HardImageBank(clf, data["Xte"], data["yte"], n_per_digit=10, seed=2)
    rng = np.random.default_rng(3)
    imgs, info = build_mosaic(puz, data["Xte"], data["yte"], rng, bank, n_hard=2)
    for algo in ("ucs", "astar", "greedy"):
        res = solve_mosaic(clf, imgs, algo, require_unique=True)
        assert res.status == "solved"
        assert res.solution is not None and all(res.solution[c] == puz[c] for c in range(81) if puz[c])
