import datetime as dt

import pandas as pd

from investos.portfolio.constraint_model import BaseConstraint
from investos.portfolio.cost_model import BaseCost


class BaseStrategy:
    """Base class for an optimization strategy.

    Must implement :py:meth:`~investos.portfolio.strategy.base_strategy.BaseStrategy.generate_trade_list` as per below.

    Attributes
    ----------
    costs : list
        Cost models evaluated during optimization strategy. Defaults to empty list. See :py:class:`~investos.portfolio.cost_model.base_cost.BaseCost` for cost model base class.
    constraints : list
        Constraints applied for optimization strategy. Defaults to empty list. See :py:class:`~investos.portfolio.constraint_model.base_constraint.BaseConstraint for optimization model base class.
    """

    def __init__(
        self,
        actual_returns: pd.DataFrame,
        costs: [BaseCost] = [],
        constraints: [BaseConstraint] = [],
        **kwargs,
    ):
        self.actual_returns = actual_returns
        self.costs = costs
        self.constraints = constraints

        self.cash_column_name = kwargs.get("cash_column_name", "cash")
        self.metadata_properties = ["cash_column_name"]

    def _zerotrade(self, holdings):
        return pd.Series(index=holdings.index, data=0.0)

    def generate_trade_list(self, holdings: pd.Series, t: dt.datetime) -> pd.Series:
        """Calculates and returns trade list (in units of currency passed in), given (added) optimization logic.

        Parameters
        ----------
        holdings : pandas.Series
            Holdings at beginning of period `t`.
        t : datetime.datetime
            The datetime for associated holdings `holdings`.
        """
        raise NotImplementedError

    def get_actual_positions_for_t(
        self, dollars_holdings: pd.Series, dollars_trades: pd.Series, t: dt.datetime
    ) -> pd.Series:
        """Calculates and returns actual positions, after accounting for trades and costs during period t."""
        dollars_holdings_plus_trades = dollars_holdings + dollars_trades

        costs = [
            cost.actual_cost(
                t,
                dollars_holdings_plus_trades=dollars_holdings_plus_trades,
                dollars_trades=dollars_trades,
            )
            for cost in self.costs
        ]

        cash_col = self.cash_column_name
        dollars_trades[cash_col] = -sum(
            dollars_trades[dollars_trades.index != cash_col]
        ) - sum(costs)
        dollars_holdings_plus_trades[cash_col] = (
            dollars_holdings[cash_col] + dollars_trades[cash_col]
        )

        dollars_holdings_at_next_t = (
            self.actual_returns.loc[t] * dollars_holdings_plus_trades
            + dollars_holdings_plus_trades
        )

        return dollars_holdings_at_next_t, dollars_trades

    def metadata_dict(self):
        meta_d = {
            "strategy": self.__class__.__name__,
        }

        if getattr(self, "risk_model", False):
            meta_d[self.risk_model.__class__.__name__] = self.risk_model.metadata_dict()

        if getattr(self, "constraints", False):
            meta_d["constraint_models"] = {
                el.__class__.__name__: el.metadata_dict() for el in self.constraints
            }

        if getattr(self, "costs", False):
            meta_d["cost_models"] = {
                el.__class__.__name__: el.metadata_dict()
                for el in self.costs
                if "Risk" not in el.__class__.__name__
            }

        return self._add_strategy_metadata_params(meta_d)

    def _add_strategy_metadata_params(self, meta_d):
        metadata_properties = getattr(self, "metadata_properties", [])
        if metadata_properties:
            meta_d["strategy_config"] = {}
            for p in metadata_properties:
                meta_d["strategy_config"][p] = getattr(self, p, "n.a.")

        return meta_d
