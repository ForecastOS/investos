import pandas as pd
import datetime as dt
import cvxpy as cvx

from investos.portfolio.constraint_model import *
from investos.portfolio.risk_model import *
from investos.portfolio.strategy import BaseStrategy
from investos.portfolio.cost_model import TradingCost, HoldingCost, BaseCost
from investos.util import values_in_time
import investos.util as util

class SPO(BaseStrategy):
    """Optimization strategy that builds trade list using single period optimization.
    """

    BASE_SOLVER_OPTS = {
        'max_iter': 50_000,
        'eps_rel': 0.0000000001,
        'eps_abs': 0.0000000001,
    }
    
    
    def __init__(self, 
                costs: [BaseCost] = [TradingCost(), HoldingCost()], 
                constraints: [BaseConstraint] = [MinWeightConstraint(), MaxWeightConstraint(), MaxLeverageConstraint(), EqualLongShortConstraint()], 
                risk_model: BaseRisk = StatFactorRisk(),
                solver=cvx.OSQP,
                solver_opts=None):
        self.forecast_returns = None # Set by Backtester in init
        self.optimizer = None # Set by Backtester in init

        self.costs = costs
        self.risk_model = risk_model
        self.constraints = constraints
        self.solver = solver
        self.solver_opts = util.deep_dict_merge(self.BASE_SOLVER_OPTS, solver_opts or {})

    
    def generate_trade_list(self, holdings: pd.Series, t: dt.datetime) -> pd.Series:
        """Calculates and returns trade list (in units of currency passed in) using convex (single period) optimization.

        Parameters
        ----------
        holdings : pandas.Series
            Holdings at beginning of period `t`.
        t : datetime.datetime
            The datetime for associated holdings `holdings`.
        """

        if t is None:
            t = dt.datetime.today()

        value = sum(holdings)
        w = holdings / value # Portfolio weights
        z = cvx.Variable(w.size)  # Portfolio trades
        wplus = w.values + z # Portfolio weights after trades

        alpha_term = cvx.sum(cvx.multiply(
            values_in_time(self.forecast_returns, t).values,
            wplus))

        assert(alpha_term.is_concave())

        costs, constraints = [], []

        for cost in self.costs:
            cost_expr, const_expr = cost.weight_expr(t, wplus, z, value)
            costs.append(cost_expr)
            constraints += const_expr

        constraints += [item for item in (con.weight_expr(t, wplus, z, value)
                                          for con in self.constraints)]

        # For help debugging: 
        
        # for el in costs:
        #     if not el.is_convex():
        #         print(t, el, "is not convex")
            # assert (el.is_convex())

        # for el in constraints:
            # if not el.is_dcp():
            #     print(t, el, "is not dcp")
            # assert (el.is_dcp())

        objective = cvx.Maximize(alpha_term - cvx.sum(costs))
        constraints += [cvx.sum(z) == 0]
        self.prob = cvx.Problem(objective, constraints) # Trades need to 0 out, i.e. cash account must adjust to make everything net to 0
        
        try:
            self.prob.solve(solver=self.solver, **self.solver_opts)

            if self.prob.status == 'unbounded':
                print(f"The problem is unbounded at {t}.")
                return self._zerotrade(holdings)

            if self.prob.status == 'infeasible':
                print(f"The problem is infeasible at {t}.")
                return self._zerotrade(holdings)

            # print("CVX problem at ", t, self.prob)
            u = pd.Series(index=holdings.index, data=(z.value * value))

            return u
        
        except (cvx.SolverError, cvx.DCPError, TypeError):
            print(f"The solver failed for {t}.")
            return self._zerotrade(holdings)
