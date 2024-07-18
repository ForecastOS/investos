import numpy as np
import pandas as pd
from investos.portfolio.risk_model.factor_utils import cov_ewa


class BiasStatsCalculator:
    """A commonly used measure to assess a risk model's accuracy"""

    def __init__(self, returns: np.ndarray, volatilities: np.ndarray) -> None:
        """
        Args:
            returns (np.ndarray): returns of K assets * T periods
            volatilities (np.ndarray): volatilities (std) of K assets * T periods
        """
        self.K, self.T = returns.shape if returns.ndim == 2 else (1, len(returns))
        self.r = returns.reshape((self.K, self.T))
        self.v = volatilities.reshape((self.K, -1))
        if self.v.shape[1] != 1 and self.v.shape[1] != self.T:
            raise ValueError("wrong shape of volatilities")

    def single_window(self, half_life: int | None = None) -> np.ndarray:
        """Calculate bias statistics, selecting entire sample period as a single window

        Args:
            half_life (int | None, optional): argument in ewa(). Defaults to None.

        Returns:
            np.ndarray: bias statistics value(s) (K * 1)
        """
        b = self.r / self.v
        b_demeaned = b - b.mean(axis=1).reshape((self.K, -1))
        B = np.sqrt(ewa(b_demeaned.T**2, half_life))
        return B.reshape((self.K, -1))

    def rolling_window(self, periods: int, half_life: int | None = None) -> np.ndarray:
        """Calculate bias statistics, specifying number of periods in rolling window

        Args:
            periods (int): number of periods in observation window
            half_life (int | None, optional): argument in ewa(). Defaults to None.

        Returns:
            np.ndarray: bias statistics values (K * (T - periods + 1)
        """
        if periods > self.T or periods < 2:
            raise ValueError("T must be between 2 and the length of returns")
        b = self.r / self.v
        b_demeaned = b - b.mean(axis=1).reshape((self.K, -1))
        B_lst = [
            np.sqrt(ewa(b_demeaned[:, t : t + periods].T ** 2, half_life))
            for t in range(self.T - periods + 1)
        ]
        return np.array(B_lst).T
