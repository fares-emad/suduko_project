"""Build the digit dataset reproducibly (fixed seed).

    python make_data.py                    # MNIST if reachable, else sklearn digits
    python make_data.py --source digits    # force the offline fallback
    python make_data.py --source mnist     # require MNIST (needs internet or data/raw/*.gz)

Outputs: data/digits19.npz and data/data_meta.json
"""
import argparse

from sudoku_ai.data import build_dataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["auto", "mnist", "digits"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build_dataset(a.source, a.seed)
