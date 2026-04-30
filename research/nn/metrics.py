"""Probabilistic forecast metrics.

All metrics follow the convention: lower is better, except `hit` (higher better).

References:
- CRPS (Continuous Ranked Probability Score): standard proper score for
  probabilistic forecasts. CRPS = 0 iff perfect prediction.
- NLL (Gaussian): assumes prediction is N(μ, σ²). Penalizes overconfidence.
"""
from __future__ import annotations
import numpy as np


SIGMA_FLOOR = 1e-4   # avoid log(0) in NLL when σ→0


def mse(y_true: float, mu: float) -> float:
    """Squared error against point prediction."""
    return float((y_true - mu) ** 2)


def mae(y_true: float, mu: float) -> float:
    return float(abs(y_true - mu))


def nll_gaussian(y_true: float, mu: float, sigma: float) -> float:
    """NLL of y under N(μ, σ²). 0.5*log(2πσ²) + (y-μ)²/(2σ²)."""
    sigma = max(float(sigma), SIGMA_FLOOR)
    return float(
        0.5 * np.log(2 * np.pi * sigma ** 2)
        + (y_true - mu) ** 2 / (2 * sigma ** 2)
    )


def crps_empirical(y_true: float, samples: np.ndarray) -> float:
    """CRPS for an empirical distribution defined by `samples`.

    CRPS(F_emp, y) = E|X - y| - 0.5 * E|X - X'|
    where X, X' are i.i.d. from the empirical distribution.

    Closed form for sorted samples (O(n log n) instead of O(n²)).
    """
    n = len(samples)
    if n == 0:
        return float('nan')
    if n == 1:
        return float(abs(samples[0] - y_true))

    s = np.sort(samples).astype(float)
    term1 = float(np.mean(np.abs(s - y_true)))
    # E|X - X'| using sorted-sample identity:
    # sum_i sum_j |s_i - s_j| = 2 * Σ_k (2k - n + 1) * s_k   (k=0..n-1)
    k = np.arange(n)
    e_xx = (2.0 / (n * n)) * float(np.sum((2 * k - n + 1) * s))
    return term1 - 0.5 * e_xx


def hit_accuracy(y_true: float, mu: float) -> float:
    """1.0 if predicted direction matches truth, 0.0 otherwise.
    Special-case y_true=0 → counts as match if mu=0."""
    if y_true == 0:
        return float(mu == 0)
    return float(np.sign(mu) == np.sign(y_true))


def hit_prob_accuracy(y_true: float, p_up: float) -> float:
    """1.0 if (p_up>0.5) matches (y_true>0). For probabilistic predictors."""
    pred_up = p_up > 0.5
    actual_up = y_true > 0
    return float(pred_up == actual_up)


def empirical_p_up(samples: np.ndarray) -> float:
    """P(y > 0) under the empirical distribution."""
    n = len(samples)
    if n == 0:
        return float('nan')
    return float(np.mean(samples > 0))


def metrics_from_samples(y_true: float, samples: np.ndarray) -> dict:
    """Convenience: compute all metrics given a true value and prediction samples."""
    samples = np.asarray(samples, dtype=float)
    if len(samples) == 0:
        return {
            'mu': np.nan, 'sigma': np.nan, 'p_up': np.nan,
            'mse': np.nan, 'mae': np.nan,
            'nll': np.nan, 'crps': np.nan,
            'hit': np.nan, 'hit_prob': np.nan,
        }
    mu = float(samples.mean())
    sigma = float(samples.std())
    p_up = empirical_p_up(samples)
    return {
        'mu':       mu,
        'sigma':    sigma,
        'p_up':     p_up,
        'mse':      mse(y_true, mu),
        'mae':      mae(y_true, mu),
        'nll':      nll_gaussian(y_true, mu, sigma),
        'crps':     crps_empirical(y_true, samples),
        'hit':      hit_accuracy(y_true, mu),
        'hit_prob': hit_prob_accuracy(y_true, p_up),
    }


__all__ = [
    'mse', 'mae', 'nll_gaussian', 'crps_empirical',
    'hit_accuracy', 'hit_prob_accuracy', 'empirical_p_up',
    'metrics_from_samples',
]
