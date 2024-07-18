from typing import Sequence
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity


def get_t_statistics(model, X, y):
    """Calculate factor t-statistics value

    Parameters
    ----------
    model : sklearn.linear_model._base.LinearRegression
        Linear regression model fits a linear model
    X : pd.DataFrame
        N*K training data with all factors value.
    y : pd.Series
        N*1 target values

    Returns
    -------
    t_values
    array
        The t-statistics for all the factors
    """
    y_hat = model.predict(X)
    residuals = y - y_hat
    residual_sum_squares = (residuals ** 2).sum()
    num_of_factors, sample_size= X.shape[1], len(y)
    root_square_error = (residual_sum_squares / (sample_size - num_of_factors - 1)) ** 0.5

    # temporary use. Current we ignore the warning from the denominator is zero
    with np.errstate(divide='ignore', invalid='ignore'):
        t_values = np.divide(model.coef_, (root_square_error / sample_size ** 0.5))
    return t_values
    
def ewa(arr: Sequence, half_life: int | None = None) -> float | np.ndarray:
    """Exponential Weighted Average (EWA)

    Parameters
    ----------
    arr : Sequence
        Array of numbers or arrays (later one on axis=0 weights more)
    half_life : int | None, optional
        Steps it takes for weight to reduce to half of the original value, by default
        None, meaning that weights are all equal

    Returns
    -------
    float | np.ndarray
        Exponential weighted average of elements in `arr`
    """
    arr = np.array(arr)
    alpha = 1.0 if half_life is None else 0.5 ** (1 / half_life)
    weights = alpha ** np.arange(len(arr) - 1, -1, -1)
    w_shape = tuple([arr.shape[0]] + [1] * (len(arr.shape) - 1))
    weights = weights.reshape(w_shape)
    sum_weight = len(arr) - 1 if half_life is None else np.sum(weights)
    return (weights * arr).sum(axis=0) / sum_weight


def calc_ewma_cov(data: np.ndarray, half_life: int | None = None, lag: int = 0) -> np.ndarray:
    """Calculate the covariance matrix as an exponential weighted average of range

    Parameters
    ----------
    data : np.ndarray
        Data matrix (K features * T periods)
    half_life : int | None, optional
        Argument in ewa(), by default None
    lag : int, optional
        Difference between to terms of fator, cov(t-lag, t), when lag is opposite, the
        result is transposed, by default 0

    Returns
    -------
    np.ndarray
        Covariance matrix
    """
    if data.shape[0] > data.shape[1]:
        raise Exception("data matrix should not have less columns than rows")
    if lag >= data.shape[1]:
        raise Exception("lag must be smaller than the number of columns of matrix")
    data = data.astype("float64")
    f_bar = data.mean(axis=1)
    data = data - f_bar.reshape(data.shape[0], -1)
    t_range = range(lag, data.shape[1]) if lag > 0 else range(data.shape[1] + lag)
    elements = np.array([np.outer(data[:, t - lag], data[:, t]) for t in t_range])
    return ewa(elements, half_life)


def num_eigvals_explain(pct: float, eigvals: np.ndarray) -> int:
    """The number of eigenvalues it takes to explain a percentage of total variance

    Parameters
    ----------
    pct : float
        Percentage of total variance
    eigvals : np.ndarray
        Eigenvalues

    Returns
    -------
    int
        Number of eigenvalues
    """
    eigvals = np.sort(eigvals)[::-1]  # descending order
    eigvals = eigvals / np.sum(eigvals)
    p, num = 0, 0
    for v in eigvals:
        p += v
        num += 1
        if p > pct:
            break
    return num


def draw_eigvals_edf(
    cov: np.ndarray,
    bandwidth: float | None = None,
    x_range: np.ndarray | None = None,
    label: str | None = None,
) -> None:
    """Draw the empirical distribution function of `cov`

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix
    bandwidth : float | None, optional
        Bandwidth of kernel density estimation (KDE), by default None
    x_range : np.ndarray | None, optional
        Range of x displayed, by default None
    label : str | None, optional
        Label on plot, by default None
    """
    eigvals = np.linalg.eigvalsh(cov).reshape(-1, 1)
    bw = np.cbrt(np.median(eigvals)) if bandwidth is None else bandwidth
    kde = KernelDensity(bandwidth=bw).fit(eigvals)
    if x_range is None:
        x = np.linspace(0, eigvals[-1] * 1.1, len(eigvals) * 10).reshape(-1, 1)
    else:
        x = x_range.reshape(-1, 1)
    probs = np.exp(kde.score_samples(x))
    plt.plot(x, probs, label=label)


def getExpWeight(window_len: int ,half_life: int ,is_unitized: bool =True):
    """Obtain the list of weights"""

    ExpWeight = (0.5**(1/half_life))**np.arange(window_len)
    if is_unitized:
        return np.ndarray(ExpWeight/np.sum(ExpWeight))
    else:
        return np.ndarray(ExpWeight)


class BiasStatsCalculator:
    """A commonly used measure to assess a risk model's accuracy"""

    def __init__(self, returns: np.ndarray, volatilities: np.ndarray) -> None:
        """
        Args:
            returns (np.ndarray): returns of K (num_of_factors) assets * (length_of_dates) periods
            volatilities (np.ndarray): volatilities (std) of num_of_factors assets * length_of_dates periods
        """
        self.num_of_factors, self.length_of_dates = returns.shape if returns.ndim == 2 else (1, len(returns))
        self.factor_returns = returns.reshape((self.num_of_factors, self.length_of_dates))
        self.factor_stds = volatilities.reshape((self.num_of_factors, -1))
        if self.factor_stds.shape[1] != 1 and self.factor_stds.shape[1] != self.length_of_dates:
            raise ValueError("wrong shape of volatilities")

    def single_window(self, half_life: int | None = 42) -> np.ndarray:
        """Calculate bias statistics, selecting entire sample period as a single window
        Args:
            half_life (int | None, optional): argument in ewa(). Defaults to None.
        Returns:
            np.ndarray: bias statistics value(s) (num_of_factors * 1)
        """
        bias = self.factor_returns / self.factor_stds
        bias_demeaned = bias - bias.mean(axis=1).reshape((self.num_of_factors, -1))
        inst_bias = np.sqrt(ewa(bias_demeaned.T**2, half_life))
        return inst_bias.reshape((self.num_of_factors, -1))

    def rolling_window(self, periods: int, half_life: int | None = None) -> np.ndarray:
        """Calculate bias statistics, specifying number of periods in rolling window
        Args:
            periods (int): number of periods in observation window
            half_life (int | None, optional): argument in ewa(). Defaults to None.

        Returns:
            np.ndarray: bias statistics values (K * (T - periods + 1)
        """
        if periods > self.length_of_dates or periods < 2:
            raise ValueError("T must be between 2 and the length of returns")
        bias = self.factor_returns / self.factor_stds
        bias_demeaned = bias - bias.mean(axis=1).reshape((self.K, -1))
        bias_demeaned_lst = [
            np.sqrt(ewa(bias_demeaned[:, t : t + periods].T ** 2, half_life))
            for t in range(self.length_of_dates - periods + 1)
        ]
        return np.array(bias_demeaned_lst).T
