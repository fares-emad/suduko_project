import numpy as np
import pytest

from sudoku_ai.data import ROOT, load_data
from sudoku_ai.nn import DigitClassifier, default_hyperparams

PUZZLE = [int(ch) for ch in
          "530070000600195000098000060800060003400803001700020006060000280000419005000080079"]
SOLUTION = [int(ch) for ch in
            "534678912672195348198342567859761423426853791713924856961537284287419635345286179"]


@pytest.fixture(scope="session")
def digits():
    data, meta = load_data()
    return data, meta


@pytest.fixture(scope="session")
def clf(digits):
    data, meta = digits
    path = ROOT / "models" / "clf.joblib"
    if path.exists():
        return DigitClassifier.load(path)
    n = min(len(data["ytr"]), 5000)
    return DigitClassifier(seed=0, **default_hyperparams(meta["img_side"])).fit(
        data["Xtr"][:n], data["ytr"][:n], data["Xva"], data["yva"])


def synthetic_probs(grid, low: dict):
    """probs for a hand-made reading. low = {cell: true_digit}: those cells are 'unsure'
    (read digit 0.50, true digit 0.45); all other given cells are confident (0.92)."""
    P = np.zeros((81, 9))
    for c, d in enumerate(grid):
        if not d:
            continue
        p = np.full(9, 0.08 / 8)
        if c in low:
            p = np.full(9, 0.05 / 7)
            p[d - 1] = 0.50
            p[low[c] - 1] = 0.45
        else:
            p[d - 1] = 0.92
        P[c] = p / p.sum()
    return P
