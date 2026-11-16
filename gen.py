"""Seeded generators: Sudoku puzzles, hard (low-confidence) digits, and image mosaics.

Everything here is GENERATED (not real data); see DATA_DICTIONARY.md.

Puzzle rule  : build a full grid by randomised backtracking, then remove cells in a
               random order, keeping a removal only if the puzzle still has exactly
               one solution, until the wanted number of givens is reached.
Mosaic rule  : a puzzle becomes an (81, n_pixels) array of images. A given digit d is
               a random test-set image of class d; a blank cell is an (almost) black
               image with tiny noise.
Hard digits  : (injection rule) for digit d take a test image of d and blend it with a
               test image of another digit e: x = (1 - a) * x_d + a * x_e. The smallest
               a on a grid 0.10, 0.15, ..., 0.90 for which the classifier reads e is
               used, provided the true digit d is still among the classifier's top 3.
               The result is a digit the network really misreads, but with low
               confidence, so the true digit stays a plausible alternative.
"""
from __future__ import annotations

import random

import numpy as np

from .csp import SudokuCSP, count_solutions


def random_full_grid(rng: random.Random) -> list[int]:
    csp = SudokuCSP([0] * 81, use_fc=True, use_ac3=False, ordering="mrv", rng=rng)
    return csp.solve().solutions[0]


def make_puzzle(solution, n_givens: int, rng: random.Random):
    """Remove cells while the solution stays unique. Returns (puzzle, solution)."""
    puzzle = list(solution)
    cells = list(range(81))
    rng.shuffle(cells)
    n_given = 81
    for c in cells:
        if n_given <= n_givens:
            break
        saved = puzzle[c]
        puzzle[c] = 0
        if count_solutions(puzzle, limit=2) == 1:
            n_given -= 1
        else:
            puzzle[c] = saved
    return puzzle, list(solution)


def generate_puzzle(n_givens: int, seed: int):
    """Seeded puzzle with a unique solution and *exactly* n_givens givens (retries seeds)."""
    for k in range(50):
        rng = random.Random(seed * 1000 + k)
        sol = random_full_grid(rng)
        puz, sol = make_puzzle(sol, n_givens, rng)
        if sum(1 for x in puz if x) == n_givens:
            return puz, sol
    raise RuntimeError(f"could not reach {n_givens} givens")


class HardImageBank:
    """Pre-computed hard digit images per true digit (see module docstring)."""

    ALPHAS = [round(0.10 + 0.05 * i, 2) for i in range(17)]

    def __init__(self, clf, pool_X, pool_y, n_per_digit=30, seed=0):
        rng = np.random.default_rng(seed)
        self.images: dict[int, list[np.ndarray]] = {d: [] for d in range(1, 10)}
        by = {d: np.where(pool_y == d)[0] for d in range(1, 10)}
        for d in range(1, 10):
            tries = 0
            while len(self.images[d]) < n_per_digit and tries < n_per_digit * 40:
                tries += 1
                xd = pool_X[rng.choice(by[d])]
                e = int(rng.choice([k for k in range(1, 10) if k != d]))
                xe = pool_X[rng.choice(by[e])]
                for a in self.ALPHAS:
                    x = (1 - a) * xd + a * xe
                    p = clf.predict_proba(x[None, :])[0]
                    if p.argmax() + 1 == e:
                        if d in (np.argsort(-p)[:3] + 1):
                            self.images[d].append(x)
                        break
        for d in range(1, 10):
            if not self.images[d]:
                raise RuntimeError(f"no hard image found for digit {d}")

    def sample(self, d: int, rng: np.random.Generator) -> np.ndarray:
        lst = self.images[d]
        return lst[int(rng.integers(len(lst)))]


def build_mosaic(puzzle, pool_X, pool_y, rng: np.random.Generator,
                 bank: HardImageBank | None = None, n_hard: int = 0,
                 hard_order=None):
    """Return (images (81, n_pixels), info). hard cells are chosen among given cells."""
    by = {d: np.where(pool_y == d)[0] for d in range(1, 10)}
    n_pix = pool_X.shape[1]
    images = np.zeros((81, n_pix))
    for c, d in enumerate(puzzle):
        if d:
            images[c] = pool_X[rng.choice(by[d])]
        else:
            images[c] = np.abs(rng.normal(0, 0.004, n_pix))  # near-black blank cell
    givens = [c for c in range(81) if puzzle[c]]
    if hard_order is None:
        hard_order = list(rng.permutation(givens))
    hard_cells = [int(c) for c in hard_order[:n_hard]]
    for c in hard_cells:
        images[c] = bank.sample(puzzle[c], rng)
    return images, {"hard_cells": hard_cells}


def mosaic_picture(images: np.ndarray, side: int) -> np.ndarray:
    """Stitch 81 images into one (9*side, 9*side) picture for plotting."""
    pic = np.zeros((9 * side, 9 * side))
    for c in range(81):
        r, k = divmod(c, 9)
        pic[r * side:(r + 1) * side, k * side:(k + 1) * side] = images[c].reshape(side, side)
    return pic
