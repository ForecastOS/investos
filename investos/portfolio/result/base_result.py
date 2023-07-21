import collections
import datetime as dt

import numpy as np
import pandas as pd

from investos.portfolio.result import SaveResult
from investos.util import clip_for_dates


class BaseResult(SaveResult):
    """The `Result` class captures portfolio data and performance for each asset and period over time.

    Instances of this object are called by the :py:meth:`investos.portfolio.controller.Controller.generate_positions` method.
    """

    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def save_data(self, name: str, t: dt.datetime, entry: pd.Series) -> None:
        """Save `entry` on `Result` object, a (pandas) Series of data as `name` for datetime `t`.

        Parameters
        ----------
        name : str
            The name `entry` is saved under in this `Result` object.
        t : datetime.datetime
            The datetime `entry` is saved under in this `Result` object.
        entry : pandas.Series
            A series of values - for a collection of assets / stocks / tickers at a specific point in time.
        """
        try:
            getattr(self, name).loc[t] = entry
        except AttributeError:
            setattr(
                self,
                name,
                (pd.Series if np.isscalar(entry) else pd.DataFrame)(
                    index=[t], data=[entry]
                ),
            )

    def save_position(self, t: dt.datetime, u: pd.Series, h_next: pd.Series) -> None:
        """
        Save data `u` and `h_next` related to position for datetime `t` on `Result` object.

        Parameters
        ----------
        t : datetime.datetime
            The datetime for associated trades `u` and t + 1 holdings `h_next`.
        u : pandas.Series
            Trades (as values) for period `t`.
        h_next : pandas.Series
            Holdings at beginning of period t + 1, after trades `u` and returns for period `t`.
        """
        self.save_data("u", t, u)
        self.save_data("h_next", t, h_next)

    @property
    def summary(self) -> None:
        """Outputs a string summary of backtest result properties and performance
        (e.g. :py:attr:`~investos.portfolio.result.base_result.BaseResult.num_periods`, :py:attr:`~investos.portfolio.result.base_result.BaseResult.sharpe_ratio`, :py:attr:`~investos.portfolio.result.base_result.BaseResult.max_drawdown`, etc.).
        """
        print(self._summary_string())

    def _summary_string(self) -> str:
        """Returns a string summary of backtest result properties and performance
        (e.g. :py:attr:`~investos.portfolio.result.base_result.BaseResult.num_periods`, :py:attr:`~investos.portfolio.result.base_result.BaseResult.sharpe_ratio`, :py:attr:`~investos.portfolio.result.base_result.BaseResult.max_drawdown`, etc.).

        Do not call directly; call :py:attr:`~investos.portfolio.result.base_result.BaseResult.summary` instead.
        """
        data = collections.OrderedDict(
            {
                "Initial timestamp": self.h.index[0],
                "Final timestamp": self.h.index[-1],
                "Total portfolio return (%)": str(round(self.total_return * 100, 2))
                + "%",
                "Annualized portfolio return (%)": str(
                    round(self.annualized_return * 100, 2)
                )
                + "%",
                "Annualized excess portfolio return (%)": str(
                    round(self.annualized_excess_return * 100, 2)
                )
                + "%",
                "Annualized excess risk (%)": str(round(self.excess_risk_annualized, 2))
                + "%",
                "Information ratio (calc. from per period returns and risk)": str(
                    round(self.information_ratio, 2)
                )
                + "x",
                "Annualized risk over risk-free (%)": str(
                    round(self.risk_over_cash_annualized, 2)
                )
                + "%",
                "Sharpe ratio (calc. from per period returns and risk)": str(
                    round(self.sharpe_ratio, 2)
                )
                + "x",
                "Max drawdown": f"{round(self.max_drawdown, 2)}%",
                "Annual turnover (x)": str(round(self.annual_turnover, 2)) + "x",
            }
        )

        return pd.Series(data=data).to_string(float_format="{:,.3f}".format)

    @property
    @clip_for_dates
    def h(self) -> pd.DataFrame:
        """Returns a pandas Dataframe of asset holdings (`h`) at the beginning of each datetime period."""
        tmp = self.h_next.copy()
        tmp = self.h_next.shift(1)  # Shift h_next to h timing
        return tmp[1:]

    @property
    @clip_for_dates
    def trades(self) -> pd.DataFrame:
        "Returns a pandas Series of trades (u)."
        return self.u

    @property
    def num_periods(self) -> int:
        """Number of periods in backtest. Note that the starting position (at t=0) does not count as a period."""
        return self.h.shape[0]

    @property
    def v(self) -> pd.Series:
        """Returns a pandas Series for the value (`v`) of the portfolio for each datetime period."""
        return self.h.sum(axis=1)

    @property
    def v_with_benchmark(self) -> pd.Series:
        """Returns a pandas Dataframe with simulated portfolio and benchmark values."""
        return pd.DataFrame({"portfolio": self.v, "benchmark": self.benchmark_v})

    @property
    def returns(self) -> pd.Series:
        """Returns a pandas Series of the returns for each datetime period (vs the previous period)."""
        val = self.v
        return pd.Series(
            data=val.values[1:] / val.values[:-1] - 1, index=val.index[1:]
        ).dropna()

    @property
    def total_return(self) -> float:
        """Returns a float representing the total return for the entire period under review."""
        return self.v[-1] / self.v[0] - 1

    @property
    def total_benchmark_return(self) -> float:
        """Returns a float representing the total return for the entire period under review."""
        return self.benchmark_v[-1] / self.benchmark_v[0] - 1

    @property
    def total_excess_return(self) -> float:
        """Returns a float representing the total return for the entire period under review."""
        return self.total_return - self.total_benchmark_return

    @property
    def annualized_return(self) -> float:
        """Returns a float representing the annualized return of the entire period under review. Uses beginning and ending portfolio values for the calculation (value @ t[-1] and value @ t[0]), as well as the number of years in the forecast."""
        return ((self.total_return + 1) ** (1 / self.years_forecast)) - 1

    @property
    def annualized_benchmark_return(self) -> float:
        """Returns a float representing the annualized benchmark return of the entire period under review. Uses beginning and ending portfolio values for the calculation (value @ t[-1] and value @ t[0]), as well as the number of years in the forecast."""
        return ((self.total_benchmark_return + 1) ** (1 / self.years_forecast)) - 1

    @property
    def excess_returns(self) -> pd.Series:
        """Returns a pandas Series of returns in excess of the benchmark."""
        return (self.returns - self.benchmark_returns).dropna()

    @property
    def returns_over_cash(self) -> pd.Series:
        """Returns a pandas Series of returns in excess of risk free returns."""
        return (self.returns - self.risk_free_returns).dropna()

    @property
    def annualized_excess_return(self) -> float:
        """Returns a float representing the annualized excess return of the entire period under review. Uses beginning and ending portfolio values for the calculation (value @ t[-1] and value @ t[0]), as well as the number of years in the forecast."""
        return self.annualized_return - self.annualized_benchmark_return

    @property
    def excess_risk_annualized(self) -> pd.Series:
        """Returns a pandas Series of risk in excess of the benchmark."""
        return self.excess_returns.std() * 100 * np.sqrt(self.ppy)

    @property
    def risk_over_cash_annualized(self) -> pd.Series:
        """Returns a pandas Series of risk in excess of the risk free rate."""
        return self.returns_over_cash.std() * 100 * np.sqrt(self.ppy)

    @property
    @clip_for_dates
    def benchmark_returns(self) -> pd.Series:
        if not hasattr(self, "benchmark"):
            self.benchmark = self.optimizer.actual["return"]["cash"]

        return self.benchmark

    @property
    @clip_for_dates
    def risk_free_returns(self) -> pd.Series:
        if not hasattr(self, "risk_free"):
            self.risk_free = self.optimizer.actual["return"]["cash"]

        return self.risk_free

    @property
    def benchmark_v(self) -> pd.Series:
        """Returns series of simulated portfolio values, if portfolio was invested 100% in benchmark at time 0"""
        benchmark_factors = self.benchmark_returns + 1
        benchmark_factors[0] = 1  # No returns for period 0

        return (
            benchmark_factors.cumprod() * self.v[0]
        )  # Calculate values if initial portfolio value was invested 100% in benchmark

    @property
    def years_forecast(self) -> float:
        """Returns a float representing the number of years in the backtest period.
        Calculated as (datetime @ t[-1] - datetime @ t[0]) / datetime.timedelta(365,0,0,0)
        """
        return (self.v.index[-1] - self.v.index[0]) / dt.timedelta(365, 0, 0, 0)

    @property
    def ppy(self) -> float:
        """Returns a float representing the number of periods per year in the backtest period.
        Calculated as :py:attr:`~investos.portfolio.result.base_result.BaseResult.num_periods` / :py:attr:`~investos.portfolio.result.base_result.BaseResult.years_forecast`
        """
        return self.num_periods / self.years_forecast

    @property
    def information_ratio(self) -> float:
        """Returns a float representing the (annualized) Information Ratio of the portfolio.

        Ratio is calculated as mean of :py:attr:`~investos.portfolio.result.base_result.base_result.BaseResult.excess_returns` / standard deviation of :py:attr:`~investos.portfolio.result.base_result.BaseResult.excess_returns`. Annualized by multiplying ratio by square root of periods per year (:py:attr:`~investos.portfolio.result.base_result.BaseResult.ppy`).
        """
        return (
            np.sqrt(self.ppy)
            * np.mean(self.excess_returns)
            / np.std(self.excess_returns)
        )

    @property
    def sharpe_ratio(self) -> float:
        """Returns a float representing the (annualized) Sharpe Ratio of the portfolio.

        Ratio is calculated as mean of :py:attr:`~investos.portfolio.result.base_result.base_result.BaseResult.excess_returns` / standard deviation of :py:attr:`~investos.portfolio.result.base_result.BaseResult.excess_returns`. Annualized by multiplying ratio by square root of periods per year (:py:attr:`~investos.portfolio.result.base_result.BaseResult.ppy`).
        """
        return (
            np.sqrt(self.ppy)
            * np.mean(self.returns_over_cash)
            / np.std(self.returns_over_cash)
        )

    @property
    def turnover(self):
        """Turnover ||u_t||_1/v_t"""
        noncash_trades = self.trades.drop(["cash"], axis=1)
        return np.abs(noncash_trades).sum(axis=1) / self.v

    @property
    def leverage(self):
        """Turnover ||u_t||_1/v_t"""
        noncash_h = self.h.drop(["cash"], axis=1)
        return np.abs(noncash_h).sum(axis=1) / self.v

    @property
    def long_leverage(self):
        """Turnover ||u_t||_1/v_t"""
        noncash_h = self.h.drop(["cash"], axis=1)
        return np.abs(noncash_h[noncash_h > 0]).sum(axis=1) / self.v

    @property
    def short_leverage(self):
        """Turnover ||u_t||_1/v_t"""
        noncash_h = self.h.drop(["cash"], axis=1)
        return np.abs(noncash_h[noncash_h < 0]).sum(axis=1) / self.v

    @property
    def annual_turnover(self):
        return self.turnover.mean() * self.ppy

    @property
    def max_drawdown(self):
        """The maximum peak to trough drawdown in percent."""
        val_arr = self.v.values
        max_dd_so_far = 0
        cur_max = val_arr[0]
        for val in val_arr[1:]:
            if val >= cur_max:
                cur_max = val
            elif (cur_max - val) / cur_max > max_dd_so_far:
                max_dd_so_far = (cur_max - val) / cur_max
        return max_dd_so_far
