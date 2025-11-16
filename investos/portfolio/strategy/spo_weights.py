import datetime as dt

import cvxpy as cvx
import pandas as pd

import investos.util as util
from investos.portfolio.constraint_model import (
    BaseConstraint,
    LongOnlyConstraint,
)
from investos.portfolio.cost_model import BaseCost
from investos.portfolio.risk_model import BaseRisk
from investos.portfolio.strategy import BaseStrategy
from investos.util import get_value_at_t


class SPOWeights(BaseStrategy):
    """Optimization strategy that builds trade list using single period optimization.

    If you're using OSQP as your solver (the default), view the following for tuning: https://osqp.org/docs/interfaces/solver_settings.html
    """

    BASE_SOLVER_OPTS = {
        "max_iter": 50_000,
    }

    def __init__(
        self,
        actual_returns: pd.DataFrame,
        target_weights: pd.DataFrame,
        costs: [BaseCost] = [],
        constraints: [BaseConstraint] = [
            LongOnlyConstraint(),
        ],
        risk_model: BaseRisk = None,
        solver=cvx.OSQP,
        solver_opts=None,
        **kwargs,
    ):
        super().__init__(
            actual_returns=actual_returns,
            costs=costs,
            constraints=constraints,
            **kwargs,
        )
        self.risk_model = risk_model
        if self.risk_model:
            self.costs.append(self.risk_model)

        self.target_weights = target_weights
        self.solver = solver
        self.solver_opts = util.deep_dict_merge(
            self.BASE_SOLVER_OPTS, solver_opts or {}
        )

        self.metadata_properties = ["solver", "solver_opts"]

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
        weights_portfolio = holdings / value  # Portfolio weights
        weights_trades = cvx.Variable(weights_portfolio.size)  # Portfolio trades
        weights_portfolio_plus_trades = (
            weights_portfolio.values + weights_trades
        )  # Portfolio weights after trades

        wdiff = (
            weights_portfolio_plus_trades
            - get_value_at_t(self.target_weights, t).values
        )

        assert wdiff.is_concave()

        # Costs not used during optimization step (since optimization is for distance from target_weights)
        constraints = []

        constraints += [
            item
            for item in (
                con.cvxpy_expression(
                    t,
                    weights_portfolio_plus_trades,
                    weights_trades,
                    value,
                    holdings.index,
                )
                for con in self.constraints
            )
        ]

        for el in constraints:
            if not el.is_dcp():
                print(t, el, "is not dcp")
            # assert el.is_dcp()

        objective = cvx.Minimize(cvx.sum(cvx.abs(wdiff)))
        constraints += [cvx.sum(weights_trades) == 0]
        self.prob = cvx.Problem(
            objective, constraints
        )  # Trades need to 0 out, i.e. cash account must adjust to make everything net to 0

        try:
            self.prob.solve(solver=self.solver, **self.solver_opts)

            if self.prob.status in ("unbounded", "infeasible"):
                print(f"The problem is {self.prob.status} at {t}.")
                return self._zerotrade(holdings)

            dollars_trades = pd.Series(
                index=holdings.index, data=(weights_trades.value * value)
            )

            return dollars_trades

        except (cvx.SolverError, cvx.DCPError, TypeError) as e:
            print(f"The solver failed for {t}. Error details: {e}")
            return self._zerotrade(holdings)
