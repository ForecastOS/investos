import pandas as pd
import datetime as dt
import copy

class BaseCost():
    """Base cost model for InvestOS. 
    Other cost models should subclass BaseCost. 
    The only requirement of custom cost models is that they (re)implement :py:meth:`~investos.portfolio.cost_model.base_cost.BaseCost.value_expr`.
    """
    
    def __init__(self):
        self.optimizer = None # Set during Optimizer initialization
        self.gamma = 1  # Can change without setting directly as: gamma * BaseCost(). Note that gamma doesn't impact actual costs in backtester / simulated performance, just trades in optimization strategy.

    
    def weight_expr(self, t, w_plus, z, value):
        cost, constraints = self._estimated_cost_for_optimization(t, w_plus, z, value)
        return self.gamma * cost, constraints


    def actual_cost(self, t: dt.datetime, h_plus: pd.Series, u: pd.Series) -> pd.Series:
        """Method that calculates per-period costs given period `t` holdings and trades.

        Parameters
        ----------
        t : datetime.datetime
            The datetime for associated trades `u` and holdings plus trades `h_plus`.
        h_plus : pandas.Series
            Holdings at beginning of period t, plus trades for period `t` (`u`). Same as `u` + `h` for `t`.
        u : pandas.Series
            Trades (as values) for period `t`.
        """
        raise NotImplementedError
    

    def __mul__(self, other):
        """Read the gamma parameter as a multiplication; so you can change self.gamma without setting it directly as: gamma * BaseCost()"""
        newobj = copy.copy(self)
        newobj.gamma *= other
        return newobj

    def __rmul__(self, other):
        """Read the gamma parameter as a multiplication; so you can change self.gamma without setting it directly as: gamma * BaseCost()"""
        return self.__mul__(other)