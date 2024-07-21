from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity


def clean_na_and_inf(df: pd.DataFrame):
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def get_t_statistics(model, X, y):
    """Calculate factor t_statistics value"""
    y_hat = model.predict(X)
    residuals = y - y_hat
    residual_sum_squares = (residuals**2).sum()
    num_of_factors, sample_size = X.shape[1], len(y)
    root_square_error = (
        residual_sum_squares / (sample_size - num_of_factors - 1)
    ) ** 0.5

    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = np.divide(model.coef_, (root_square_error / sample_size**0.5))
    return t_values


def calc_exp_weighted_avg(
    arr: Sequence, half_life: int | None = None
) -> float | np.ndarray:
    """Calcualted exponential weighted average"""
    arr = np.array(arr)
    alpha = 1.0 if half_life is None else 0.5 ** (1 / half_life)
    weights = alpha ** np.arange(len(arr) - 1, -1, -1)
    w_shape = tuple([arr.shape[0]] + [1] * (len(arr.shape) - 1))
    weights = weights.reshape(w_shape)
    sum_weight = len(arr) - 1 if half_life is None else np.sum(weights)
    return (weights * arr).sum(axis=0) / sum_weight


def calc_exp_weighted_moving_avg_cov(
    data: np.ndarray, half_life: int | None = None, lag: int = 0
) -> np.ndarray:
    """Calculate the covariance matrix as an exponential weighted average of range"""
    if data.shape[0] > data.shape[1]:
        raise Exception("data matrix should not have less columns than rows")
    if lag >= data.shape[1]:
        raise Exception("lag must be smaller than the number of columns of matrix")
    data = data.astype("float64")
    f_bar = data.mean(axis=1)
    data = data - f_bar.reshape(data.shape[0], -1)
    t_range = range(lag, data.shape[1]) if lag > 0 else range(data.shape[1] + lag)
    elements = np.array([np.outer(data[:, t - lag], data[:, t]) for t in t_range])
    return calc_exp_weighted_avg(elements, half_life)


def draw_eigvals_edf(
    cov: np.ndarray,
    bandwidth: float | None = None,
    x_range: np.ndarray | None = None,
    label: str | None = None,
) -> None:
    """Draw the empirical distribution function of `cov`"""
    eigvals = np.linalg.eigvalsh(cov).reshape(-1, 1)
    bw = np.cbrt(np.median(eigvals)) if bandwidth is None else bandwidth
    kde = KernelDensity(bandwidth=bw).fit(eigvals)
    if x_range is None:
        x = np.linspace(0, eigvals[-1] * 1.1, len(eigvals) * 10).reshape(-1, 1)
    else:
        x = x_range.reshape(-1, 1)
    probs = np.exp(kde.score_samples(x))
    plt.plot(x, probs, label=label)


def get_exp_weight(window_len: int, half_life: int, is_unitized: bool = True):
    """Obtain the list of weights"""

    ExpWeight = (0.5 ** (1 / half_life)) ** np.arange(window_len)
    if is_unitized:
        return np.ndarray(ExpWeight / np.sum(ExpWeight))
    else:
        return np.ndarray(ExpWeight)


class BiasStatsCalculator:
    """Measure to assess a risk model's accuracy"""

    def __init__(self, returns: np.ndarray, stds: np.ndarray) -> None:
        self.num_of_factors, self.length_of_dates = (
            returns.shape if returns.ndim == 2 else (1, len(returns))
        )
        self.factor_returns = returns.reshape(
            (self.num_of_factors, self.length_of_dates)
        )
        self.factor_stds = stds.reshape((self.num_of_factors, -1))
        if (
            self.factor_stds.shape[1] != 1
            and self.factor_stds.shape[1] != self.length_of_dates
        ):
            raise ValueError("wrong shape of standard deviation")

    def apply_single_window(self, half_life: int | None = 42) -> np.ndarray:
        """Calculate bias statistics, selecting entire sample period as a single window"""
        bias = self.factor_returns / self.factor_stds
        factor_bias_stats = np.sqrt(np.mean((bias) ** 2, axis=0)).reshape(
            self.length_of_dates, 1
        )
        factor_vol_multiplier = np.sqrt(
            calc_exp_weighted_avg(factor_bias_stats**2, half_life)
        )
        return factor_vol_multiplier

    def apply_rolling_window(
        self, periods: int, half_life: int | None = None
    ) -> np.ndarray:
        """Calculate bias statistics, specifying number of periods in rolling window"""
        if periods > self.length_of_dates or periods < 2:
            raise ValueError("T must be between 2 and the length of returns")
        bias = self.factor_returns / self.factor_stds
        bias_demeaned = bias - bias.mean(axis=1).reshape((self.K, -1))
        bias_demeaned_lst = [
            np.sqrt(
                calc_exp_weighted_avg(
                    bias_demeaned[:, t : t + periods].T ** 2, half_life
                )
            )
            for t in range(self.length_of_dates - periods + 1)
        ]
        return np.array(bias_demeaned_lst).T
