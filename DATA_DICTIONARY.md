# Data dictionary

Real data vs generated data is marked on every item. Nothing in this project uses
gender, age, region, disability or any personal attribute (the dataset has none).

## 1. Digit images (REAL) - `data/digits19.npz`, made by `make_data.py`

| Array | Shape | Meaning | Real / generated |
|---|---|---|---|
| `Xtr`, `Xva`, `Xte` | (n, 784) for MNIST or (n, 64) for the fallback | Flattened grey-scale digit image, pixels scaled to [0, 1] | real |
| `ytr`, `yva`, `yte` | (n,) | Digit label 1..9 (digit 0 is dropped because Sudoku has no 0) | real |

`data/data_meta.json` records which source was used (`mnist` or `sklearn_digits`),
the image side (28 or 8), split sizes and the seed.

**Sources and splits**

* **MNIST** (preferred): 50,000 train / 10,000 validation taken from the 60,000 training
  images with a seeded permutation; the official 10,000 test images are untouched;
  zeros dropped afterwards. Needs internet (or the four `.gz` files in `data/raw/`).
* **scikit-learn digits** (automatic fallback when MNIST is unreachable): the 1,797
  8x8 handwritten digits bundled with scikit-learn (UCI optical-digits test set).
  Zeros dropped, then a stratified seeded split: 64% train / 16% validation / 20% test.
  **The results delivered with this project were produced with this fallback.**

## 2. Network outputs (derived)

| Item | Meaning |
|---|---|
| `predict_proba(X)` | (n, 9) probabilities, column j = digit j+1, temperature-scaled on the validation set |
| `T` (in `models/clf.joblib`, `results/nn_metrics.json`) | the single temperature parameter |

## 3. Generated Sudoku puzzles (GENERATED) - `sudoku_ai/gen.py`

| Item | Meaning | How it is made |
|---|---|---|
| `puzzle` | list of 81 ints, 0 = blank | full grid by randomised backtracking (seeded), then cells removed in random order, keeping a removal only if exactly one solution remains, until the wanted number of givens (25 / 30 / 35) |
| `solution` | list of 81 ints | the full grid before removal |
| seed | integer | `generate_puzzle(n_givens, seed)`; each team/run uses its own seed |

## 4. Generated image mosaics (GENERATED from real digit images)

| Item | Meaning | How it is made |
|---|---|---|
| `images` | (81, n_pixels) array, row-major cells | a given digit d -> random **test-set** image of class d; a blank -> a near-black image (`abs(N(0, 0.004))` per pixel) |
| blank rule | cell is blank if mean pixel < 0.03 | brightness rule, not the network |
| hard digits (injection rule) | `x = (1-a)*x_d + a*x_e` | x_d: test image of the true digit d; x_e: test image of another digit e. Smallest a in {0.10, 0.15, ..., 0.90} for which the network reads e, kept only if d is still in the network's top 3. Built once per seed in `HardImageBank` |
| `hard_cells` | which given cells carry a hard digit | first k cells of a seeded random order of the givens (nested over k = 0..4 so levels are comparable) |
| "misread level" | number of injected hard digits | the *actual* number of misreads can be higher when the network also misreads an ordinary digit by itself; the experiment tables record `n_misread` in the raw CSVs |

This is **not** photographed real-world Sudoku: it is a mosaic of separately written
digits.

## 5. Search quantities (DERIVED)

| Item | Definition |
|---|---|
| correction | change one given cell from the read digit to another digit |
| step cost | `max(0, log p(read) - log p(new))` using the calibrated probabilities |
| candidate cells | the 12 least confident given cells (default) |
| candidate digits | the 3 most probable alternatives per cell (default) |
| `max_corrections` | 3 |

## 6. Result files

| File | Content |
|---|---|
| `results/nn_metrics.json` | accuracy, NLL, ECE of the network (raw and calibrated) vs the logistic-regression baseline |
| `results/corrections_raw_{solvable,unique}.csv` | one row per (seed, puzzle, misread level, method): solved, recovered, corrections, cost, nodes expanded/generated, goal tests, time |
| `results/csp_raw.csv` | one row per (seed, givens, puzzle, configuration): the CSP counters |
| `results/exp1_*.{csv,md,png}`, `exp2.*`, `exp3_*.*` | aggregated tables (mean ± std over seeds) and plots |
