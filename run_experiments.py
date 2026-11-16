"""Run the three experiments, write tables (CSV + markdown) and plots to results/.

    python run_experiments.py                 # 5 seeds x 20 puzzles (100 puzzles per level)
    python run_experiments.py --quick         # 2 seeds x 4 puzzles, for a fast check
    python run_experiments.py --seeds 5 --puzzles 20 --givens 30
"""
import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sudoku_ai.data import ROOT, load_data
from sudoku_ai.eval import (SEARCH_ALGOS, correction_experiment_seed, csp_experiment_seed,
                            fmt_table, mean_std_over_seeds)
from sudoku_ai.nn import DigitClassifier

OUT = ROOT / "results"
COLORS = {"ucs": "#1f77b4", "greedy": "#ff7f0e", "astar": "#2ca02c",
          "none": "#7f7f7f", "random": "#d62728",
          "BT": "#9467bd", "BT+FC": "#17becf", "BT+FC+AC-3": "#8c564b"}


def grouped_bars(ax, agg, x_col, series_col, y, title, ylabel, log=False):
    xs = sorted(agg[x_col].unique())
    series = list(dict.fromkeys(agg[series_col]))
    w = 0.8 / len(series)
    for j, s in enumerate(series):
        sub = agg[agg[series_col] == s].set_index(x_col).reindex(xs)
        ax.bar(np.arange(len(xs)) + j * w - 0.4 + w / 2, sub[y + "_mean"], w,
               yerr=sub[y + "_std"], capsize=2, label=s, color=COLORS.get(s))
    ax.set_xticks(range(len(xs)), xs)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--puzzles", type=int, default=20, help="puzzles per seed")
    ap.add_argument("--givens", type=int, default=30, help="givens for experiments 1 and 3")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.puzzles = 2, 4
    seeds = list(range(1, a.seeds + 1))
    OUT.mkdir(exist_ok=True)
    data, meta = load_data()
    clf = DigitClassifier.load(ROOT / "models" / "clf.joblib")
    levels = [0, 1, 2, 3, 4]
    t0 = time.time()

    # ---- experiments 1 and 3 (two goal definitions) --------------------------------
    for mode, unique in (("solvable", False), ("unique", True)):
        rows = Parallel(n_jobs=a.jobs)(
            delayed(correction_experiment_seed)(clf, data, s, a.puzzles, levels, a.givens, unique)
            for s in seeds)
        df = pd.DataFrame([r for rs in rows for r in rs])
        df.to_csv(OUT / f"corrections_raw_{mode}.csv", index=False)
        for c in ("solved", "recovered", "limit_hit"):
            df[c] = df[c].astype(float)

        # Experiment 1
        d1 = df[df.method.isin(SEARCH_ALGOS)]
        vals = ["nodes_expanded", "nodes_generated", "goal_tests", "cost", "recovered", "time_s"]
        agg1 = mean_std_over_seeds(d1, ["level", "method"], vals)
        agg1.to_csv(OUT / f"exp1_{mode}.csv", index=False)
        (OUT / f"exp1_{mode}.md").write_text(fmt_table(agg1, ["level", "method"], vals, 3))
        # cost of solved runs only (failures have no cost)
        fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
        grouped_bars(axs[0], agg1, "level", "method", "nodes_expanded",
                     "Exp. 1: nodes expanded", "nodes expanded (mean ± std over seeds)", log=True)
        grouped_bars(axs[1], agg1, "level", "method", "cost", "Exp. 1: cost of correction set",
                     "total cost (solved runs)")
        grouped_bars(axs[2], agg1, "level", "method", "recovered",
                     "Exp. 1: exact-solution recovery", "recovery rate")
        fig.suptitle(f"UCS vs Greedy vs A* (goal = {mode}), {a.seeds} seeds x {a.puzzles} puzzles")
        fig.tight_layout()
        fig.savefig(OUT / f"exp1_{mode}.png", dpi=130)
        plt.close(fig)

        # Experiment 3
        d3 = df[df.method.isin(["ucs", "none", "random"])].copy()
        d3["method"] = d3.method.replace({"ucs": "nn_guided (UCS)", "none": "no correction",
                                          "random": "random correction"})
        agg3 = mean_std_over_seeds(d3, ["level", "method"], ["solved", "recovered"])
        agg3.to_csv(OUT / f"exp3_{mode}.csv", index=False)
        (OUT / f"exp3_{mode}.md").write_text(
            fmt_table(agg3, ["level", "method"], ["solved", "recovered"], 3))
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        grouped_bars(ax, agg3, "level", "method", "recovered",
                     f"Exp. 3: exact-solution recovery (goal = {mode})", "recovery rate")
        fig.tight_layout()
        fig.savefig(OUT / f"exp3_{mode}.png", dpi=130)
        plt.close(fig)
        print(f"[{mode}] done after {time.time() - t0:.0f}s")

    # ---- experiment 2 ----------------------------------------------------------------
    rows2 = Parallel(n_jobs=a.jobs)(
        delayed(csp_experiment_seed)(s, a.puzzles, [25, 30, 35]) for s in seeds)
    df2 = pd.DataFrame([r for rs in rows2 for r in rs])
    df2.to_csv(OUT / "csp_raw.csv", index=False)
    df2["cutoff"] = (df2.status == "cutoff").astype(float)
    df2["correct"] = df2["correct"].astype(float)
    vals2 = ["assignments", "backtracks", "fc_prunings", "ac3_prunings", "consistency_checks",
             "time_s", "cutoff", "correct"]
    agg2 = mean_std_over_seeds(df2, ["givens", "config"], vals2)
    agg2.to_csv(OUT / "exp2.csv", index=False)
    (OUT / "exp2.md").write_text(fmt_table(agg2, ["givens", "config"], vals2, 3))
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    grouped_bars(axs[0], agg2, "givens", "config", "assignments", "Exp. 2: assignments tried",
                 "assignments (mean ± std)", log=True)
    grouped_bars(axs[1], agg2, "givens", "config", "backtracks", "Exp. 2: backtracks",
                 "backtracks (mean ± std)", log=True)
    grouped_bars(axs[2], agg2, "givens", "config", "time_s", "Exp. 2: run time", "seconds",
                 log=True)
    fig.tight_layout()
    fig.savefig(OUT / "exp2.png", dpi=130)
    plt.close(fig)
    print(f"all done in {time.time() - t0:.0f}s; results in {OUT}")


if __name__ == "__main__":
    main()
