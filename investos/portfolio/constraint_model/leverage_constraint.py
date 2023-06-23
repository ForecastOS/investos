import cvxpy as cvx

from investos.util import values_in_time
from investos.portfolio.constraint_model.base_constraint import BaseConstraint

class MaxLeverageConstraint(BaseConstraint):
    """
    A constraint that enforces a limit on the (absolute) leverage of the portfolio.

    E.g. For leverage of 2.0x, a portfolio with 100MM net value 
    (i.e. the portfolio value if it were converted into cash, 
    ignoring liquidation / trading costs) 
    could have 200MM of (combined long and short) exposure.

    Parameters
    ----------
    limit : float, optional
        The minimum weight of each asset in the portfolio. Defaults to -0.05.

    **kwargs :
        Additional keyword arguments.
    """
    
    def __init__(self, limit: float = 2.0, **kwargs):
        self.limit = limit

    def weight_expr(self, t, w_plus, z, v):
        """
        Returns a series of holding constraints.

        Parameters
        ----------
        t : datetime
            The current time.

        w_plus : series
            Portfolio weights after trades z.

        z : series
            Trades for period t

        v : float
            Value of portfolio at period t

        Returns
        -------
        series
            The holding constraints based on the portfolio leverage after trades.
        """
        return cvx.sum(cvx.abs(w_plus[:-1])) <= self.limit