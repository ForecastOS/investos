import datetime as dt

import pandas as pd


class BaseStrategy:
    """Base class for an optimization strategy.

    Must implement :py:meth:`~investos.portfolio.strategy.base_strategy.BaseStrategy.generate_trade_list` as per below.

    Attributes
    ----------
    costs : list
        Cost models evaluated during optimization strategy. Defaults to empty list. See :py:class:`~investos.portfolio.cost_model.base_cost.BaseCost` for cost model base class.
    constraints : list
        Constraints applied for optimization strategy. Defaults to empty list. See [TBU] for optimization model base class.
    """

    def __init__(self):
        self.costs = []
        self.constraints = []
        self.risk_model = None

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

    def metadata_dict(self):
        return {
            "strategy": self.__class__.__name__,
            "risk_model": self.risk_model
            and {self.risk_model.__class__.__name__: self.risk_model.metadata_dict()},
            "constraint_models": {
                el.__class__.__name__: el.metadata_dict() for el in self.constraints
            },
            "cost_models": {
                el.__class__.__name__: el.metadata_dict()
                for el in self.costs
                if "Risk" not in el.__class__.__name__
            },
        }
