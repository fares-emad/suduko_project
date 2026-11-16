"""Digit data: MNIST when it can be obtained, otherwise scikit-learn's bundled digits.

Output of build_dataset(): data/digits19.npz with
  Xtr, ytr, Xva, yva, Xte, yte    pixels in [0, 1] (flattened), labels 1..9 (digit 0 dropped)
and data/data_meta.json describing the source, image side and split sizes.
"""
from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
NPZ = DATA_DIR / "digits19.npz"
META = DATA_DIR / "data_meta.json"

MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _fetch_mnist_file(name: str) -> bytes:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    local = RAW_DIR / name
    if local.exists():
        return local.read_bytes()
    last = None
    for base in MNIST_MIRRORS:
        try:
            with urllib.request.urlopen(base + name, timeout=15) as r:
                blob = r.read()
            local.write_bytes(blob)
            return blob
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"could not download {name}: {last}")


def load_mnist_raw():
    blobs = {k: _fetch_mnist_file(v) for k, v in MNIST_FILES.items()}
    img = lambda b: np.frombuffer(gzip.decompress(b), np.uint8, offset=16).reshape(-1, 784)  # noqa: E731
    lab = lambda b: np.frombuffer(gzip.decompress(b), np.uint8, offset=8)  # noqa: E731
    return (img(blobs["train_images"]) / 255.0, lab(blobs["train_labels"]),
            img(blobs["test_images"]) / 255.0, lab(blobs["test_labels"]))


def _drop_zero(X, y):
    keep = y != 0
    return X[keep], y[keep].astype(int)


def build_dataset(source: str = "auto", seed: int = 0, verbose: bool = True) -> dict:
    """Create data/digits19.npz. source: 'auto' | 'mnist' | 'digits'."""
    rng = np.random.default_rng(seed)
    meta = {"seed": seed}
    used = None
    if source in ("auto", "mnist"):
        try:
            Xa, ya, Xte, yte = load_mnist_raw()
            used = "mnist"
        except Exception as e:  # noqa: BLE001
            if source == "mnist":
                raise
            if verbose:
                print(f"[make_data] MNIST not reachable ({e}); falling back to sklearn digits.")
    if used == "mnist":
        perm = rng.permutation(len(Xa))
        tr, va = perm[:50000], perm[50000:]  # 50,000 train / 10,000 validation
        Xtr, ytr = _drop_zero(Xa[tr], ya[tr])
        Xva, yva = _drop_zero(Xa[va], ya[va])
        Xte, yte = _drop_zero(Xte, yte)
        meta.update(source="mnist", img_side=28,
                    note="MNIST, official test set untouched; digit 0 dropped")
    else:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split

        d = load_digits()
        X, y = _drop_zero(d.data / 16.0, d.target)
        Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
        Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=0.1875, stratify=ytv,
                                              random_state=seed)
        meta.update(source="sklearn_digits", img_side=8,
                    note=("scikit-learn bundled 8x8 handwritten digits (UCI optical digits, "
                          "1797 images); used because MNIST was not downloadable. "
                          "Digit 0 dropped; stratified 64/16/20 split"))
    meta.update(n_train=len(ytr), n_val=len(yva), n_test=len(yte), n_pixels=int(Xtr.shape[1]))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NPZ, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte)
    META.write_text(json.dumps(meta, indent=2))
    if verbose:
        print("[make_data]", json.dumps(meta))
    return meta


def load_data(build_if_missing: bool = True):
    if not NPZ.exists():
        if not build_if_missing:
            raise FileNotFoundError("run make_data.py first")
        build_dataset()
    z = np.load(NPZ)
    meta = json.loads(META.read_text())
    return {k: z[k] for k in z.files}, meta
