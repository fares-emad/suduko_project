# P7 - Handwritten Sudoku Reader and Solver (NN + Search + CSP)

A neural network reads a Sudoku written in handwritten digits. When it misreads a digit
the puzzle breaks. **Search** (UCS / Greedy / A*) finds the most likely set of
corrections, and a **CSP** (Backtracking, Forward Checking, AC-3) tests each corrected
puzzle and solves it.

```
mosaic of digit images
   -> blank cells found by a brightness rule
   -> NN reads the other cells and gives probabilities
   -> AC-3 on the reading: conflict?
   -> Search proposes corrections, cheapest first   (cost = how much less likely the new digit is)
   -> CSP checks each corrected grid; first solvable (or uniquely solvable) one wins
   -> otherwise "UNREADABLE" + list of the least confident cells
```

## IMPORTANT: which digit data was used

The sandbox this project was built in could not download MNIST (network blocked), and
neither PyTorch nor TensorFlow was installed. So:

* `make_data.py` **tries MNIST first** and **falls back automatically to scikit-learn's
  bundled 8x8 handwritten digits** (1,797 images). All delivered results come from the
  fallback. The network is a scikit-learn `MLPClassifier`, so no deep-learning package is
  needed.
* On your own computer with internet, run `python make_data.py --source mnist` and then
  everything else; the code switches to 28x28 images by itself. **Re-run the experiments
  and re-read the numbers**, they will change (the network will be more accurate).

## Setup and run commands

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # pinned versions

python make_data.py                  # data/digits19.npz   (seed 0; MNIST if reachable)
python train_nn.py                   # models/clf.joblib + results/nn_*.{json,png}
python -m pytest -q                  # 45 tests
python run_demo.py                   # integrated demo + results/demo_*.png
python run_experiments.py            # 3 experiments, 5 seeds x 20 puzzles (~1.5 min, 2 cores)
python run_experiments.py --quick    # fast check
```

Demo options: `python run_demo.py --seed 7 --givens 30 --misreads 2 --algo astar --unique`

## Layout (maps to the required deliverables)

| Path | Content |
|---|---|
| `make_data.py`, `sudoku_ai/data.py` | reproducible data script (fixed seed) |
| `DATA_DICTIONARY.md` | every item, real vs generated, how generated parts are made |
| `sudoku_ai/nn.py`, `train_nn.py` | network, temperature scaling, ECE, logistic-regression baseline |
| `sudoku_ai/csp.py` | CSP: BT, FC, AC-3 (+ optional MAC) as switches, with counters |
| `sudoku_ai/search.py` | UCS, Greedy, A* with counters; admissible heuristic |
| `sudoku_ai/gen.py` | seeded puzzle generator, hard-digit bank, mosaics |
| `sudoku_ai/integration.py` | the pipeline and the two baselines (no correction, random) |
| `sudoku_ai/eval.py`, `run_experiments.py` | the three experiments, mean ± std tables, plots |
| `run_demo.py` | integrated demo |
| `tests/` | 45 tests (normal, edge, impossible) |
| `results/` | metrics, tables (`.md`/`.csv`), plots, raw CSVs, `experiments.log` |

## Design choices

* **Search state** = set of (cell, new digit) corrections, at most one per cell, at most 3
  in total. Candidates: the 12 least confident given cells x their 3 best alternative
  digits. Step cost = `max(0, log p(read) - log p(new))`.
* **Goal test** = the corrected grid is solvable (minimum version) or has *exactly one*
  solution (`--unique`, the optional extension). A fast duplicate-digit scan rejects
  obviously broken grids before the CSP is called; the CSP then uses FC + AC-3.
* **A\* heuristic** (admissible): pick conflicting pairs that share no cell; each needs a
  different correction, so sum the cheapest correction cost of each pair. Verified by a
  brute-force test. States that cannot be repaired within the remaining corrections get
  h = infinity and are pruned (Greedy and A* only). A* may re-open states because the
  heuristic is not proven consistent.
* **CSP** uses minimum-remaining-values variable ordering in all three configurations so
  that plain BT stays feasible (a cutoff of 100,000 assignments is reported, never
  hidden). Plain BT computes legal digits from assigned peers (counter
  `consistency_checks`); FC keeps domains.
* **Hard digits**: MNIST-style digits are rarely misread, so misreads are injected by
  blending two digits (rule in `DATA_DICTIONARY.md`). The misread digits are *really*
  misread by the network, but the true digit stays in its top 3.

## Results (this run: fallback 8x8 digits, 5 seeds x 20 puzzles, 30 givens)

Full tables: `results/exp*.md`. Plots: `results/exp*.png`, `results/demo_*.png`.

**Network** (`results/nn_metrics.json`): test accuracy 97.8% vs 96.3% for logistic
regression; NLL 0.086 vs 0.172; ECE 0.013 vs 0.075. Temperature scaling did **not**
improve test calibration here (raw ECE 0.007) because the validation set has only 243
images; reported as is.

**Experiment 1 - UCS vs Greedy vs A\*** (goal = unique; mean over 5 seeds)

| misread level | nodes expanded UCS | Greedy | A* | exact recovery UCS = A* | Greedy |
|---|---|---|---|---|---|
| 0 | 13.2 | 2.1 | 2.5 | 0.96 | 0.95 |
| 1 | 94.6 | 7.9 | 5.2 | 0.93 | 0.82 |
| 2 | 582 | 22.1 | 13.1 | 0.76 | 0.61 |
| 3 | 2059 | 42.3 | 29.6 | 0.49 | 0.39 |
| 4 | 4903 | 59.6 | 67.0 | 0.00 | 0.00 |

UCS and A* always return the same (optimal) cost; A* needs one to two orders of magnitude
fewer nodes. Greedy is cheap but pays a higher correction cost and recovers less often.
Level 4 cannot be recovered because the limit is 3 corrections ("UNREADABLE" is the
intended answer). The costs shown for level 4 come only from runs that ended with a
solvable (but wrong) grid.

**Experiment 2 - BT vs BT+FC vs BT+FC+AC-3** (25/30/35 givens): mean assignments
336 / 276 / 276 (25 givens), 92 / 83 / 83 (30), 49 / 48 / 48 (35); no cutoffs, all
solutions correct. FC cuts assignments, backtracks and run time. AC-3 as a one-off
pre-processing step removes ~300 digits but, **with MRV ordering, does not reduce
assignments any further** and costs a little extra time on these easy puzzles; its value
here is the early conflict detection used by the pipeline. This is an honest finding, not
a bug.

**Experiment 3 - NN-guided vs no correction vs random correction** (exact true-solution
recovery, goal = unique):

| misread level | NN-guided (UCS) | no correction | random (500 tries) |
|---|---|---|---|
| 0 | 0.96 | 0.57 | 0.70 |
| 1 | 0.93 | 0.00 | 0.11 |
| 2 | 0.76 | 0.00 | 0.00 |
| 3 | 0.49 | 0.00 | 0.00 |
| 4 | 0.00 | 0.00 | 0.00 |

The "misread level" is the number of *injected* hard digits; the network also misreads
about 0.5 ordinary digits per puzzle on this small dataset (see `n_misread` in the raw
CSV), which is why level 0 is not 100%.

## Limits (be honest in the report)

* **Not MNIST**: 8x8 images, a tiny test set (324 images), one train/val/test split.
* **One trained network.** The 5 seeds vary puzzles, mosaics, hard-digit banks and the
  random baseline, but the network is trained once (`models/clf.joblib`). Retraining it
  per seed would add network variance.
* **Wrong but valid solutions.** A misread digit may not break any rule; the puzzle can
  stay solvable (even uniquely) with a wrong grid. Exact recovery therefore can be below
  100% even when the search "succeeds". The `unique` goal reduces but does not remove this.
* **Candidate pool**: the true misread cell must be among the 12 least confident cells and
  the true digit among the 3 best alternatives; otherwise it cannot be found.
* **Hard digits are synthetic** blends, not real hard handwriting.
* Mosaics are separately written digits, not photographs of real Sudoku.
* Search optimality is optimality in the *network's* probabilities, not in truth.

## Still to do by you (not generated here)

Report (max 12 pages), the 10-minute presentation with the 5-minute live demo, and the
individual contribution statements plus the git log (`git init` and commit regularly; this
folder is not a git repository yet).
