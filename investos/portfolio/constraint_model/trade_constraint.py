import numpy as np
import cvxpy as cvx

from investos.util import values_in_time
from investos.portfolio.constraint_model.base_constraint import BaseConstraint

class MaxTradeConstraint(BaseConstraint):
    """
    A constraint that enforces a limit on the (absolute) max trade size (as a weight, or fraction, of daily volume).

    Parameters
    ----------
    limit : float, optional
        The limit on the (absolute) max trade size (as a weight, or fraction, of volume for period `t`).

    **kwargs :
        Additional keyword arguments.
    """
    
    def __init__(self, limit: float = 1.0, **kwargs):
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
            The holding constraints based on the max trade constraint.
        """
        return cvx.abs(z[:-1]) * v <= np.array(
                values_in_time(self.optimizer.forecast['volume'], t) *
                values_in_time(self.optimizer.forecast['price'], t)
            ) * self.limit