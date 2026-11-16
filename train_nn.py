"""Train the digit network and compare it with a logistic-regression baseline.

    python train_nn.py

Outputs: models/clf.joblib, results/nn_metrics.json, results/nn_confusion_matrix.png,
         results/nn_reliability.png
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from sudoku_ai.data import ROOT, load_data
from sudoku_ai.nn import (DigitClassifier, default_hyperparams, expected_calibration_error,
                          make_baseline, nll, reliability_curve)

SEED = 0


def main():
    data, meta = load_data()
    Xtr, ytr, Xva, yva, Xte, yte = (data[k] for k in ("Xtr", "ytr", "Xva", "yva", "Xte", "yte"))
    print(f"data source: {meta['source']}  train/val/test = {len(ytr)}/{len(yva)}/{len(yte)}")

    clf = DigitClassifier(seed=SEED, **default_hyperparams(meta["img_side"]))
    clf.fit(Xtr, ytr, Xva, yva)
    (ROOT / "models").mkdir(exist_ok=True)
    clf.save(ROOT / "models" / "clf.joblib")

    base = make_baseline(SEED).fit(Xtr, ytr)
    yi = yte - 1
    P_raw, P_cal, P_lr = clf.predict_proba_raw(Xte), clf.predict_proba(Xte), base.predict_proba(Xte)
    metrics = {
        "data_source": meta["source"], "temperature": clf.T,
        "network": {"test_accuracy": float((P_cal.argmax(1) == yi).mean()),
                    "nll_raw": nll(P_raw, yi), "nll_calibrated": nll(P_cal, yi),
                    "ece_raw": expected_calibration_error(P_raw, yi),
                    "ece_calibrated": expected_calibration_error(P_cal, yi)},
        "logistic_regression_baseline": {
            "test_accuracy": float((P_lr.argmax(1) == yi).mean()),
            "nll": nll(P_lr, yi), "ece": expected_calibration_error(P_lr, yi)},
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "nn_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    cm = confusion_matrix(yte, P_cal.argmax(1) + 1, labels=list(range(1, 10)))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(9), range(1, 10))
    ax.set_yticks(range(9), range(1, 10))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Test confusion matrix (network)")
    for i in range(9):
        for j in range(9):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out / "nn_confusion_matrix.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, P in (("raw", P_raw), ("temperature-scaled", P_cal), ("logistic reg.", P_lr)):
        xs, ys, _ = reliability_curve(P, yi)
        ax.plot(xs, ys, "o-", ms=3, label=name)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability diagram (test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "nn_reliability.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
