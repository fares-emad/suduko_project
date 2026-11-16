"""Digit classifier (small neural network) with temperature-scaled probabilities.

The probabilities, not just the arg-max, matter here: the search uses
-log p(digit) as the price of changing a cell, so they should be calibrated.
Temperature scaling fits one number T on the validation set:
    p_cal = softmax(log(p_raw) / T)
"""
from __future__ import annotations

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def _softmax_T(logp, T):
    z = logp / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(proba, y_idx):
    p = np.clip(proba[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)
    return float(-np.log(p).mean())


def expected_calibration_error(proba, y_idx, n_bins=10):
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_idx).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_curve(proba, y_idx, n_bins=10):
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_idx).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
            ns.append(int(m.sum()))
    return xs, ys, ns


class DigitClassifier:
    """Predicts digits 1..9. predict_proba returns an (n, 9) array; column j = digit j+1."""

    def __init__(self, hidden=(64,), seed=0, max_iter=300, alpha=1e-3):
        self.hidden, self.seed, self.max_iter, self.alpha = hidden, seed, max_iter, alpha
        self.mlp = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=max_iter,
                                 random_state=seed)
        self.T = 1.0

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        self.mlp.fit(Xtr, ytr)
        if Xva is not None:
            self._fit_temperature(Xva, yva)
        return self

    def _fit_temperature(self, Xva, yva):
        logp = np.log(np.clip(self.mlp.predict_proba(Xva), 1e-12, 1.0))
        y_idx = np.asarray(yva) - 1
        res = minimize_scalar(lambda T: nll(_softmax_T(logp, T), y_idx),
                              bounds=(0.2, 6.0), method="bounded")
        self.T = float(res.x)

    def predict_proba_raw(self, X):
        return self.mlp.predict_proba(X)

    def predict_proba(self, X):
        logp = np.log(np.clip(self.mlp.predict_proba(X), 1e-12, 1.0))
        return _softmax_T(logp, self.T)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1) + 1

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def default_hyperparams(img_side: int) -> dict:
    if img_side >= 28:
        return dict(hidden=(128,), max_iter=25, alpha=1e-4)
    return dict(hidden=(64,), max_iter=400, alpha=1e-3)


def make_baseline(seed=0):
    return LogisticRegression(max_iter=2000, random_state=seed)
