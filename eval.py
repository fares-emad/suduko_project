"""Experiment helpers: the three experiments, aggregation (mean +/- std over seeds), plots.

Experiment 1  UCS vs Greedy vs A*          (nodes, cost, recovery) for 0-4 injected misreads
Experiment 2  BT vs BT+FC vs BT+FC+AC-3    (assignments, backtracks, time) for 25/30/35 givens
Experiment 3  NN-guided vs none vs random  (exact-solution recovery) for 0-4 injected misreads

Every seed uses its own puzzles, mosaics and hard-image bank. The network itself is
trained once (models/clf.joblib); see README "Limits".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .csp import SudokuCSP
from .gen import HardImageBank, build_mosaic, generate_puzzle
from .integration import (read_mosaic, solve_random_correction, solve_reading,
                          solve_without_correction)

SEARCH_ALGOS = ["ucs", "greedy", "astar"]


# --------------------------------------------------------------- experiments 1 and 3
def correction_experiment_seed(clf, data, seed, n_puzzles, levels, n_givens,
                               require_unique, node_limit=20000, random_budget=500):
    bank = HardImageBank(clf, data["Xte"], data["yte"], n_per_digit=30, seed=seed)
    rows = []
    for i in range(n_puzzles):
        puz, sol = generate_puzzle(n_givens, seed=seed * 1000 + i)
        rng = np.random.default_rng(seed * 1000 + i)
        givens = [c for c in range(81) if puz[c]]
        order = list(rng.permutation(givens))  # nested: level k uses the first k cells
        for k in levels:
            imgs, info = build_mosaic(puz, data["Xte"], data["yte"],
                                      np.random.default_rng(seed * 100000 + i * 10 + k),
                                      bank, n_hard=k, hard_order=order)
            grid, probs = read_mosaic(clf, imgs)
            base = dict(seed=seed, puzzle=i, level=k,
                        n_misread=sum(1 for c in givens if grid[c] != puz[c]))
            for algo in SEARCH_ALGOS:
                res = solve_reading(grid, probs, algo, require_unique=require_unique,
                                    node_limit=node_limit)
                s = res.search
                rows.append({**base, "method": algo, "solved": res.status == "solved",
                             "recovered": res.solution == sol,
                             "n_corrections": len(res.corrections),
                             "cost": s.total_cost if s.found else np.nan,
                             "nodes_expanded": s.nodes_expanded,
                             "nodes_generated": s.nodes_generated,
                             "goal_tests": s.goal_tests, "time_s": s.time_s,
                             "limit_hit": s.status == "limit"})
            ok, solution = solve_without_correction(grid, require_unique)
            rows.append({**base, "method": "none", "solved": ok, "recovered": solution == sol,
                         "n_corrections": 0})
            ok, solution, tries = solve_random_correction(
                grid, np.random.default_rng(seed * 7 + i * 13 + k), random_budget, 3,
                require_unique)
            rows.append({**base, "method": "random", "solved": ok, "recovered": solution == sol,
                         "goal_tests": tries})
    return rows


# -------------------------------------------------------------------- experiment 2
CSP_CONFIGS = {"BT": dict(), "BT+FC": dict(use_fc=True),
               "BT+FC+AC-3": dict(use_fc=True, use_ac3=True)}


def csp_experiment_seed(seed, n_puzzles, givens_levels, max_assignments=100000):
    rows = []
    for g in givens_levels:
        for i in range(n_puzzles):
            puz, sol = generate_puzzle(g, seed=seed * 1000 + i + g * 100000)
            for name, cfg in CSP_CONFIGS.items():
                csp = SudokuCSP(puz, ordering="mrv", max_assignments=max_assignments, **cfg)
                r = csp.solve()
                rows.append({"seed": seed, "givens": g, "puzzle": i, "config": name,
                             "status": r.status, "correct": bool(r.solved and r.solutions[0] == sol),
                             **r.stats.as_dict()})
    return rows


# ---------------------------------------------------------------------- aggregation
def mean_std_over_seeds(df: pd.DataFrame, group_cols, value_cols) -> pd.DataFrame:
    """Average within each seed first, then mean and std (ddof=1) across seeds."""
    per_seed = df.groupby(["seed"] + group_cols)[value_cols].mean().reset_index()
    g = per_seed.groupby(group_cols)[value_cols]
    mean, std = g.mean(), g.std(ddof=1)
    out = mean.add_suffix("_mean").join(std.add_suffix("_std"))
    return out.reset_index()


def fmt_table(agg: pd.DataFrame, group_cols, value_cols, digits=2) -> str:
    cols = group_cols + value_cols
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in agg.iterrows():
        cells = [str(r[c]) for c in group_cols]
        for v in value_cols:
            cells.append(f"{r[v + '_mean']:.{digits}f} ± {r[v + '_std']:.{digits}f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
