import statistics

import pandas as pd

from investos.portfolio.result import BaseResult


class BacktestController:
    """Container class that runs backtests using passed-in portfolio engineering `strategy` (see :py:class:`~investos.portfolio.strategy.base_strategy.BaseStrategy`), then saves results into `result` (see :py:class:`~investos.backtest.result.Result`) class."""

    def __init__(self, strategy, **kwargs):
        self.strategy = strategy
        self.strategy.backtest_controller = self

        # Optional
        self._set_time_periods(**kwargs)

        self.hooks = kwargs.get("hooks", {})
        self.initial_portfolio = kwargs.get(
            "initial_portfolio",
            self._create_initial_portfolio_if_not_provided(**kwargs),
        )

        # Create results instance for saving performance
        self.results = kwargs.get("results_model", BaseResult)(
            start_date=self.start_date,
            end_date=self.end_date,
        )
        self.results.strategy = self.strategy

    def generate_positions(self):
        print("Generating historical portfolio trades and positions...")

        # Create t == 0 position (no trades)
        t = self._get_initial_t()
        u = pd.Series(index=self.initial_portfolio.index, data=0)
        h_next = self.initial_portfolio  # Includes cash
        self.results.save_position(t, u, h_next)

        # Walk through time and calculate future trades, estimated and actual costs and returns, and resulting positions
        for t in self.time_periods:
            u = self.strategy.generate_trade_list(h_next, t)
            h_next, u = self.strategy.get_actual_positions_for_t(h_next, u, t)
            self.results.save_position(t, u, h_next)

            for func in self.hooks.get("after_trades", []):
                func(self, t, u, h_next)

        print("Done simulating.")
        return self.results

    def _set_time_periods(self, **kwargs):
        time_periods = kwargs.get("time_periods", self.strategy.actual_returns.index)
        self.start_date = kwargs.get("start_date", time_periods[0])
        self.end_date = kwargs.get("end_date", time_periods[-1])

        self.time_periods = time_periods[
            (time_periods >= self.start_date) & (time_periods <= self.end_date)
        ]

    def _create_initial_portfolio_if_not_provided(self, **kwargs):
        aum = kwargs.get("aum", 100_000_000)
        initial_portfolio = pd.Series(
            index=self.strategy.actual_returns.columns, data=0
        )
        initial_portfolio[self.strategy.cash_column_name] = aum

        return initial_portfolio

    def _get_initial_t(self):
        median_time_delta = statistics.median(
            self.time_periods[1:5] - self.time_periods[0:4]
        )

        return pd.to_datetime(self.start_date) - median_time_delta
