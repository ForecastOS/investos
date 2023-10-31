import pandas as pd

from investos.portfolio.result.base_result import BaseResult


class WeightsResult(BaseResult):
    """For generating backtest results from portfolio weights and historical returns.

    In this model, trades for each period happen after returns.
    """

    def __init__(
        self,
        initial_weights,
        trade_weights,
        actual_returns,
        aum=100_000_000,
        *args,
        **kwargs,
    ):
        self.set_h_next(initial_weights, trade_weights, actual_returns, aum)
        self.risk_free = kwargs.get(
            "risk_free", pd.Series(0.0, index=actual_returns.index)
        )
        self.benchmark = kwargs.get(
            "benchmark", pd.Series(0.0, index=actual_returns.index)
        )
        start_date = kwargs.get("start_date", trade_weights.index[0])
        end_date = kwargs.get("end_date", trade_weights.index[-1])
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            actual_returns=actual_returns,
            **kwargs,
        )

    def set_h_next(self, initial_weights, trade_weights, returns, aum):
        print(
            "Calculating holding values from trades, returns, and initial weights and AUM..."
        )

        h = initial_weights
        for t in trade_weights.index:
            r = returns.loc[t]
            u = trade_weights.loc[t]
            h_next = (h + u) * (1.0 + r)
            self.save_position(t, u, h_next)
            h = h_next

        self.u *= aum
        self.h_next *= aum

        print("Done calculations.")
