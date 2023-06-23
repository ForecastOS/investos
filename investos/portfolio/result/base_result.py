import pandas as pd
import numpy as np

import datetime as dt
import collections

class BaseResult():
    """The `Result` class captures portfolio data and performance for each asset and period over time.

    Instances of this object are called by the :py:meth:`investos.portfolio.controller.Controller.generate_positions` method.
    """

    def __init__(self):
        pass

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
            setattr(self, name,
                    (pd.Series if np.isscalar(entry) else
                     pd.DataFrame)(index=[t], data=[entry]))


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
        data = collections.OrderedDict({
            'Number of periods':
                self.num_periods,
            'Initial timestamp':
                self.h.index[0],
            'Final timestamp':
                self.h.index[-1],
            'Annualized portfolio return (%)':
                str(round(self.annualized_return * 100, 2)) + '%',
            'Total portfolio return (%)':
                str(round(self.total_return * 100, 2)) + '%',
            'Sharpe ratio':
                self.sharpe_ratio,
            'Max drawdown':
                f"{round(self.max_drawdown, 2)}%",
            'Annual turnover (x)':
                str(round(self.turnover.mean() * self.ppy, 2)) + 'x',
        })

        return (pd.Series(data=data).
                to_string(float_format='{:,.3f}'.format))


    @property
    def h(self) -> pd.DataFrame:
        """Returns a pandas Dataframe of asset holdings (`h`) at the beginning of each datetime period.
        """
        tmp = self.h_next.copy()
        tmp = self.h_next.shift(1) # Shift h_next to h timing
        return tmp[1:]


    @property
    def num_periods(self) -> int:
        """Number of periods in backtest. Note that the starting position (at t=0) does not count as a period."""
        return self.h.shape[0]


    @property
    def v(self) -> pd.Series:
        """Returns a pandas Series for the value (`v`) of the portfolio for each datetime period.
        """
        return self.h.sum(axis=1)


    @property
    def returns(self) -> pd.Series:
        """Returns a pandas Series of the returns for each datetime period (vs the previous period)."""
        val = self.v
        return pd.Series(data=val.values[1:] / val.values[:-1] - 1,
                         index=val.index[1:]).dropna()


    @property
    def total_return(self) -> float:
        """Returns a float representing the total return for the entire period under review.
        """
        return self.v[-1] / self.v[0] - 1
    

    @property
    def annualized_return(self) -> float:
        """Returns a float representing the annualized return of the entire period under review. Uses beginning and ending portfolio values for the calculation (value @ t[-1] and value @ t[0]), as well as the number of years in the forecast.
        """
        return (
            ( (self.total_return + 1) ** (1 / self.years_forecast) ) - 1
        )


    @property
    def excess_returns(self) -> pd.Series:
        """Returns a pandas Series of returns in excess of the (cash) benchmark.
        """
        return (self.returns - self.optimizer.actual['return']['cash']).dropna()


    @property
    def years_forecast(self) -> float:
        """Returns a float representing the number of years in the backtest period.
        Calculated as (datetime @ t[-1] - datetime @ t[0]) / datetime.timedelta(365,0,0,0)
        """
        return (self.v.index[-1] - self.v.index[0]) / dt.timedelta(365,0,0,0)


    @property
    def ppy(self) -> float:
        """Returns a float representing the number of periods per year in the backtest period.
        Calculated as :py:attr:`~investos.portfolio.result.base_result.BaseResult.num_periods` / :py:attr:`~investos.portfolio.result.base_result.BaseResult.years_forecast`
        """
        return self.num_periods / self.years_forecast


    @property
    def sharpe_ratio(self) -> float:
        """Returns a float representing the (annualized) 
        `Sharpe Ratio <https://en.wikipedia.org/wiki/Sharpe_ratio>`_ 
        of the portfolio.

        Ratio is calculated as mean of :py:attr:`~investos.portfolio.result.base_result.base_result.BaseResult.excess_returns` / standard deviation of :py:attr:`~investos.portfolio.result.base_result.BaseResult.excess_returns`. Annualized by multiplying ratio by square root of periods per year (:py:attr:`~investos.portfolio.result.base_result.BaseResult.ppy`).
        
        TBU: accept benchmark for long-only portfolios / portfolios tracking benchmark
        """
        return (
            np.sqrt(self.ppy) * np.mean(self.excess_returns) /
            np.std(self.excess_returns)
        )


    @property
    def turnover(self):
        """Turnover ||u_t||_1/v_t
        """
        noncash_trades = self.u.drop(['cash'], axis=1)
        return np.abs(noncash_trades).sum(axis=1) / self.v


    @property
    def max_drawdown(self):
        """The maximum peak to trough drawdown in percent.
        """
        val_arr = self.v.values
        max_dd_so_far = 0
        cur_max = val_arr[0]
        for val in val_arr[1:]:
            if val >= cur_max:
                cur_max = val
            elif 100 * (cur_max - val) / cur_max > max_dd_so_far:
                max_dd_so_far = 100 * (cur_max - val) / cur_max
        return max_dd_so_far