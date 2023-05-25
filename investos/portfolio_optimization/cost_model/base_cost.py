import pandas as pd
import datetime as dt

class BaseCost():
    """Base cost model for InvestOS. 
    Other cost models should subclass BaseCost. 
    The only requirement of custom cost models is that they (re)implement :py:meth:`~investos.portfolio_optimization.cost_model.base_cost.BaseCost.value_expr`.
    """
    
    def __init__(self):
        pass

    
    def value_expr(self, t: dt.datetime, h_plus: pd.Series, u: pd.Series) -> pd.Series:
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