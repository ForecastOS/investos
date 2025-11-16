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
        self.set_dollars_holdings_at_next_t(
            initial_weights, trade_weights, actual_returns, aum
        )
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

    def set_dollars_holdings_at_next_t(
        self, initial_weights, trade_weights, returns, aum
    ):
        print(
            "Calculating holding values from trades, returns, and initial weights and AUM..."
        )

        dollars_holdings = initial_weights
        for t in trade_weights.index:
            r = returns.loc[t]
            dollars_trades = trade_weights.loc[t]
            dollars_holdings_at_next_t = (dollars_holdings + dollars_trades) * (1.0 + r)
            self.save_position(t, dollars_trades, dollars_holdings_at_next_t)
            dollars_holdings = dollars_holdings_at_next_t

        self.dollars_trades *= aum
        self.dollars_holdings_at_next_t *= aum

        print("Done calculations.")
