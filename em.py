from collections import defaultdict
from itertools import combinations
from typing import List
from time import time
import os

import numpy as np

from comb import comb
from imagemodel import ImageModel

eps = 1e-6

def labelspace(n, y):
    """
    Generate label space H for a bag of `n` instances, `y` of which are
    positive.
    """
    assert n >= y, 'n={} must be greater than y={}'.format(n, y)

    H = np.zeros((comb(n, y), n), 'float32')
    for i, c in enumerate(combinations(range(n), y)):
        H[i, list(c)] = True
    return H


def compute_alpha(rho, H: np.ndarray) -> np.ndarray:
    """
    Given vector rho of `f(X_i^j)` and `y^j`, compute new values
    `alpha_j^h` for all `h` in `H^j`.
    """
    # Precompute `p(h|X)` for all `h`
    g = np.apply_along_axis(lambda h: np.prod(rho * h + (1 - rho) * (1 - h)),
                            1, H)

    # Compute `p(h|X) = p(y,h|X)/p(y|X)`
    return g / np.sum(g)


def compute_phi(alpha: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Given vector of `alpha_j(h)` for all `h` from labelspace `H`,
    compute vector of `Phi_i^j`.
    """
    # We utilize the fact that h_i^j is binary
    return np.dot(H.T, alpha)

def bag_likelihood(rho, alphas, H):
    p = np.clip(H * rho + (1 - H) * (1 - rho), eps, 1 - eps)
    alphas = np.clip(alphas, eps, 1 - eps)
    b = np.sum(np.log(p), axis=1) - np.log(alphas)
    return np.dot(alphas, b)

def likelihood(rhos, alphas, Hs):
    m = len(rhos)
    L = sum(bag_likelihood(rho, alpha, H)
            for rho, alpha, H in zip(rhos, alphas, Hs))
    return L / m


class EMAlgorithm:

    def __init__(self, Xs: List, ys: List, f: ImageModel, model_dir=None):
        assert len(Xs) == len(ys), 'Xs and ys must have the same length'

        self.Xs = Xs
        self.ys = ys
        self.f = f
        self.model_dir = model_dir

        # These are `H^j`
        self.Hs = [labelspace(X.shape[0], y) for X, y in zip(Xs, ys)]

        # Allocate arrays for the rhos
        self.rhos = [self.f(X) for X in Xs]

        # Allocate arrays for alphas and Phis
        self.alphas = [compute_alpha(rho, H) for rho, H in zip(self.rhos, self.Hs)]
        self.Phis = [np.zeros(X.shape[0]) for X in Xs]

    def fit(self, niters, metrics=[]):
        """
        Fit the model.

        Parameters
        ----------
        metrics : List[Metric], optional
            A list of metrics to be computed after each iteration.

        Returns
        -------
        dict
            a dictionary containing lists of the computed metrics
        """
        history = defaultdict(list)

        iter = 0

        self._update_history(history, metrics, likelihood(self.rhos,
            self.alphas, self.Hs))

        while iter < niters:
            iter += 1

            print("I#{}".format(iter))

            start_time = time()

            self._expectation()
            self._maximization()

            end_time = time()

            L = likelihood(self.rhos, self.alphas, self.Hs)
            self._update_history(history, metrics, L)
            self._save_model()

            print("└ finished in {:.2f}s: L={:.4E}".format(end_time - start_time, L))

        return dict(history)

    def _expectation(self):
        m = len(self.Xs)

        # Update `alpha_j(h)` for all bags
        for j in range(m):
            alpha = compute_alpha(self.rhos[j], self.Hs[j])
            self.alphas[j] = alpha

        # Update the weights Phi for all bags
        for j in range(m):
            self.Phis[j] = compute_phi(self.alphas[j], self.Hs[j])

    def _maximization(self):
        m = len(self.Xs)

        # Update the image model using the fresh-computed weights
        self.f.fit(np.concatenate(self.Phis), 1)

        # Compute the differences to previous iteration
        for j in range(m):
            self.rhos[j] = self.f(self.Xs[j])

    def _update_history(self, history, metrics, likelihood):
        history["likelihood"].append(likelihood)

        for metric in metrics:
            metric_name = type(metric).__name__.lower()
            history[metric_name].append(metric.evaluate(self))

    def _save_model(self):
        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)
            model_path = os.path.join(self.model_dir, "model.out")
            self.f.save(model_path)

            print("│ model saved to [{}]".format(model_path))

