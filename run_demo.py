"""One integrated demo: python run_demo.py

Shows, for one seeded puzzle with injected misreads:
  1. the handwritten input mosaic and what the network read,
  2. AC-3 finding the conflict,
  3. the search trace (UCS / Greedy / A*) with counters,
  4. the correction log and the solved grid,
  5. CSP pruning with and without FC / AC-3,
  6. the system with the network ON versus OFF.
Figures go to results/demo_*.png.

Options: --seed 7 --givens 30 --misreads 2 --algo ucs|greedy|astar --unique
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from sudoku_ai.csp import SudokuCSP, ac3_detects_conflict, cell_name, find_conflicts
from sudoku_ai.data import ROOT, load_data
from sudoku_ai.gen import HardImageBank, build_mosaic, generate_puzzle, mosaic_picture
from sudoku_ai.integration import read_mosaic, solve_reading, solve_without_correction
from sudoku_ai.nn import DigitClassifier

OUT = ROOT / "results"


def grid_str(g, marks=()):
    lines = []
    for r in range(9):
        if r in (3, 6):
            lines.append("------+-------+------")
        row = []
        for c in range(9):
            i = r * 9 + c
            ch = str(g[i]) if g[i] else "."
            row.append(f"{ch}*" if i in marks else ch)
        cells = [x.ljust(1) for x in row]
        lines.append(" ".join(" ".join(cells[k:k + 3]) + (" |" if k < 6 else "")
                              for k in range(0, 9, 3)))
    return "\n".join(lines)


def draw_grid(ax, g, title, marks=None, color="crimson", fixed=None):
    ax.set_xlim(0, 9)
    ax.set_ylim(9, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    for i in range(10):
        lw = 2.0 if i % 3 == 0 else 0.5
        ax.plot([0, 9], [i, i], "k", lw=lw)
        ax.plot([i, i], [0, 9], "k", lw=lw)
    for c in range(81):
        r, k = divmod(c, 9)
        if marks and c in marks:
            ax.add_patch(Rectangle((k, r), 1, 1, fill=True, alpha=0.3, color=color))
        if g[c]:
            ax.text(k + .5, r + .55, str(g[c]), ha="center", va="center", fontsize=12,
                    color="black" if (fixed is None or c in fixed) else "tab:blue")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--givens", type=int, default=30)
    ap.add_argument("--misreads", type=int, default=2)
    ap.add_argument("--algo", default="ucs", choices=["ucs", "greedy", "astar"])
    ap.add_argument("--unique", action="store_true", help="require exactly one solution")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    data, meta = load_data()
    clf = DigitClassifier.load(ROOT / "models" / "clf.joblib")
    side = meta["img_side"]
    puzzle, solution = generate_puzzle(a.givens, a.seed)
    bank = HardImageBank(clf, data["Xte"], data["yte"], n_per_digit=30, seed=a.seed)
    imgs, info = build_mosaic(puzzle, data["Xte"], data["yte"], np.random.default_rng(a.seed),
                              bank, n_hard=a.misreads)
    grid, probs = read_mosaic(clf, imgs)
    wrong = [c for c in range(81) if grid[c] != puzzle[c]]

    print(f"=== INPUT  (data source: {meta['source']}, seed {a.seed}) ===")
    print(f"true puzzle: {sum(1 for x in puzzle if x)} givens; injected hard digits at "
          f"{[cell_name(c) for c in info['hard_cells']]}")
    print("\nWhat the network read (* = differs from the true digit):")
    print(grid_str(grid, marks=set(wrong)))
    print(f"\nactual misreads: {[(cell_name(c), puzzle[c], grid[c]) for c in wrong]}")

    # 2. AC-3 on the first reading
    ac = ac3_detects_conflict(grid)
    print("\n=== AC-3 ON THE FIRST READING ===")
    if ac["conflict"]:
        print(f"conflict detected -> {ac['reason']}  (AC-3 revisions: {ac['stats'].ac3_revisions})")
    else:
        print("AC-3 found no empty domain (the misread digits do not break a rule directly)")

    # 3-4. search
    res = solve_reading(grid, probs, a.algo, require_unique=a.unique)
    s = res.search
    print(f"\n=== SEARCH ({a.algo.upper()}) ===")
    print(f"expanded={s.nodes_expanded}  generated={s.nodes_generated}  goal tests={s.goal_tests}"
          f"  (CSP calls={s.csp_calls}, rejected by quick conflict scan={s.quick_rejects})")
    print("first expansions (g = cost so far):")
    for t in s.trace[:8]:
        st = ", ".join(f"{cell_name(c)}->{d}" for c, d in t["state"]) or "(first reading)"
        h = "" if t["h"] is None else f", h={t['h']:.2f}"
        print(f"  #{t['n']:<3} g={t['g']:.3f}{h}  {st}")
    print(f"\nRESULT: {res.status.upper()} - {res.message}")
    for row in res.correction_log():
        print(f"  correction: {row['cell']}  read {row['read_digit']} -> {row['new_digit']}"
              f"  (cost {row['cost']})")
    if res.solution:
        print("\nSolved grid:")
        print(grid_str(res.solution))
        print(f"matches the true solution: {res.solution == solution}")

    # 5. CSP pruning on the corrected puzzle
    fixed = list(grid)
    for c in res.corrections:
        fixed[c.cell] = c.new
    base_grid = fixed if res.solution else puzzle
    print("\n=== CSP PRUNING (same puzzle, three configurations) ===")
    print(f"{'config':<12}{'assign':>8}{'backtr':>8}{'FC prun':>9}{'AC3 prun':>10}{'checks':>9}")
    stats = {}
    for name, cfg in (("BT", {}), ("BT+FC", dict(use_fc=True)),
                      ("BT+FC+AC-3", dict(use_fc=True, use_ac3=True))):
        r = SudokuCSP(base_grid, **cfg).solve()
        st = r.stats
        stats[name] = st
        print(f"{name:<12}{st.assignments:>8}{st.backtracks:>8}{st.fc_prunings:>9}"
              f"{st.ac3_prunings:>10}{st.consistency_checks:>9}")

    # 6. NN on vs off
    ok, sol0 = solve_without_correction(grid, a.unique)
    print("\n=== NETWORK-GUIDED CORRECTION  ON vs OFF ===")
    print(f"OFF (accept first reading): {'solved' if ok else 'FAILED - no solution'}"
          f"{'  (but wrong!)' if ok and sol0 != solution else ''}")
    print(f"ON  ({a.algo.upper()} corrections):    {res.status}, "
          f"exact true solution = {res.solution == solution}")

    # ---- figure 1: mosaic, reading, result ----
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
    axs[0].imshow(mosaic_picture(imgs, side), cmap="gray_r")
    axs[0].set_title("1. input mosaic (hard digits boxed)", fontsize=9)
    axs[0].axis("off")
    for c in info["hard_cells"]:
        r, k = divmod(c, 9)
        axs[0].add_patch(Rectangle((k * side - .5, r * side - .5), side, side, fill=False,
                                   ec="crimson", lw=1.5))
    draw_grid(axs[1], grid, "2. network's first reading (red = misread)", marks=set(wrong),
              fixed=set(range(81)))
    corr_cells = {c.cell for c in res.corrections}
    draw_grid(axs[2], res.solution or grid,
              "3. CSP solution after search corrections" if res.solution else "3. UNREADABLE",
              marks=corr_cells, color="gold", fixed={c for c in range(81) if puzzle[c]})
    fig.tight_layout(pad=2.0)
    fig.savefig(OUT / "demo_pipeline.png", dpi=130)
    plt.close(fig)

    # ---- figure 2: search trace + CSP pruning ----
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    tr = s.trace
    axs[0].plot([t["n"] for t in tr], [t["g"] for t in tr], "o-", ms=3, label="g (cost so far)")
    if tr and tr[0]["h"] is not None:
        axs[0].plot([t["n"] for t in tr], [t["g"] + t["h"] for t in tr], "s--", ms=3,
                    label="g + h")
    axs[0].set_xlabel("expansion #")
    axs[0].set_title(f"search trace ({a.algo.upper()}, first {len(tr)} expansions)", fontsize=10)
    axs[0].legend(fontsize=8)
    names = list(stats)
    axs[1].bar(names, [stats[n].assignments for n in names], color=["#9467bd", "#17becf", "#8c564b"])
    axs[1].set_title("CSP assignments tried", fontsize=10)
    ac_base = ac3_detects_conflict(base_grid)
    axs[2].bar(["before AC-3", "after AC-3"],
               [sum(ac_base["domain_sizes_before"]), sum(ac_base["domain_sizes_after"])],
               color=["#bbbbbb", "#2ca02c"])
    axs[2].set_title("total domain size: AC-3 pruning", fontsize=10)
    fig.tight_layout(pad=2.0)
    fig.savefig(OUT / "demo_search_csp.png", dpi=130)
    plt.close(fig)
    print(f"\nfigures: {OUT / 'demo_pipeline.png'}  {OUT / 'demo_search_csp.png'}")


if __name__ == "__main__":
    main()
