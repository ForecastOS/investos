import datetime as dt
from datetime import datetime

import cvxpy as cvx
import numpy as np
import pandas as pd
from dask import compute, delayed

import investos.util as util
from investos.portfolio.constraint_model import (
    BaseConstraint,
    LongOnlyConstraint,
    MaxWeightConstraint,
)
from investos.portfolio.cost_model import BaseCost
from investos.portfolio.risk_model import BaseRisk
from investos.portfolio.strategy import BaseStrategy
from investos.util import _solve_and_extract_z, values_in_time


class SPOTranches(BaseStrategy):
    """Optimization strategy that builds trade list using single period optimization for n_periods_held number of tranches.

    Each tranch is sold after n_periods_held.

    If you're using OSQP as your solver (the default), view the following for tuning: https://osqp.org/docs/interfaces/solver_settings.html
    """

    BASE_SOLVER_OPTS = {
        "max_iter": 50_000,
    }

    def __init__(
        self,
        actual_returns: pd.DataFrame,
        forecast_returns: pd.DataFrame,
        costs: [BaseCost] = [],
        constraints: [BaseConstraint] = [
            LongOnlyConstraint(),
            MaxWeightConstraint(),
        ],
        risk_model: BaseRisk = None,
        solver=cvx.OSQP,
        solver_opts=None,
        n_periods_held=5,
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

        self.forecast_returns = forecast_returns
        self.solver = solver
        self.solver_opts = util.deep_dict_merge(
            self.BASE_SOLVER_OPTS, solver_opts or {}
        )

        self.metadata_properties = ["solver", "solver_opts"]

        self.n_periods_held = n_periods_held
        self.u_unwind = {}

        self.polishing = kwargs.get("polishing", True)
        self.polishing_denom = kwargs.get("polishing_denom", 100_000)

        self.discreet_shares = kwargs.get("discreet_shares", False)
        if self.discreet_shares:
            self.n_share_block = kwargs.get("n_share_block", 100)
            self.actual_prices = kwargs.get("actual_prices", None)

    def formulate_optimization_problem(self, holdings: pd.Series, t: dt.datetime):
        value = sum(holdings)
        w = holdings / value  # Portfolio weights
        z = cvx.Variable(w.size)  # Portfolio trades
        wplus = w.values + z  # Portfolio weights after trades

        # JUST CURRENT TRANCHE FOR SPO_TRANCHES
        alpha_term = cvx.sum(
            cvx.multiply(values_in_time(self.forecast_returns, t).values, z)
        )

        assert alpha_term.is_concave()

        costs, constraints = [], []

        for cost in self.costs:
            cost_expr, const_expr = cost.weight_expr(
                t, z, z, value, holdings.index
            )  # z only: only calc costs per tranche, no wplus for tranche opt class
            costs.append(cost_expr)
            constraints += const_expr

        constraints += [
            item
            for item in (
                con.weight_expr(t, wplus, z, value, holdings.index)
                for con in self.constraints
            )
        ]

        # For help debugging:
        for el in costs:
            if not el.is_convex():
                print(t, el, "is not convex")
            # assert el.is_convex()

        for el in constraints:
            if not el.is_dcp():
                print(t, el, "is not dcp")
            # assert el.is_dcp()

        objective = cvx.Maximize(alpha_term - cvx.sum(costs))
        constraints += [cvx.sum(z) == 0]
        prob = cvx.Problem(
            objective, constraints
        )  # Trades need to 0 out, i.e. cash account must adjust to make everything net to 0

        return (prob, z)

    def precompute_trades_distributed(self, holdings: pd.Series, time_periods):
        delayed_tasks = []
        z_variables = []

        for t in time_periods:
            prob, z = self.formulate_optimization_problem(holdings, t)
            z_variables.append(z)
            delayed_task = delayed(_solve_and_extract_z)(
                prob, z, t, self.solver, self.solver_opts
            )
            delayed_tasks.append(delayed_task)

        print(f"\nComputing trades at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
        results = compute(*delayed_tasks)
        print(
            f"\nFinished computing trades at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
        )

        print("\nSaving distributed trades...")
        for t, z_value in results:
            self._save_data("z_distr", t, z_value)
        print("\nDistributed trades saved.")

        return results

    def generate_trade_list(self, holdings: pd.Series, t: dt.datetime) -> pd.Series:
        """Calculates and returns trade list (in units of currency passed in) using convex (single period) optimization.

        Parameters
        ----------
        holdings : pandas.Series
            Holdings at beginning of period `t`.
        t : datetime.datetime
            The datetime for associated holdings `holdings`.
        """
        is_distributed = self.backtest_controller.distributed

        if t is None:
            t = dt.datetime.today()

        if is_distributed:
            z = self.z_distr.loc[t].values

        else:
            prob, z = self.formulate_optimization_problem(holdings, t)

            try:
                prob.solve(solver=self.solver, **self.solver_opts)

                if prob.status == "unbounded":
                    print(f"The problem is unbounded at {t}.")
                    return self._zerotrade(holdings)

                if prob.status == "infeasible":
                    print(f"The problem is infeasible at {t}.")
                    return self._zerotrade(holdings)

                z = z.value

            except (cvx.SolverError, cvx.DCPError, TypeError) as e:
                print(f"The solver failed for {t}. Error details: {e}")
                return self._zerotrade(holdings)

        value = sum(holdings)
        u = pd.Series(index=holdings.index, data=(z * value))

        # Zero out small values; cash (re)calculated later based on trade balance, cash value here doesn't matter
        if self.polishing:
            u[abs(u) < value / self.polishing_denom] = 0

        # Round trade to discreet n_share_block (default: 100)
        if self.discreet_shares:
            prices = values_in_time(self.actual_prices, t)
            block_prices = prices * self.n_share_block
            block_prices[self.cash_column_name] = None

            non_cash_mask = u.index != self.cash_column_name
            u[non_cash_mask] = (
                u[non_cash_mask] / block_prices[non_cash_mask]
            ).round() * block_prices[non_cash_mask]

        # Unwind logic starts
        trades_saved = self.backtest_controller.results.u.shape[0]
        self._save_data("u_unwind_pre", t, u)

        if trades_saved >= self.n_periods_held:
            # Use holdings_unwind, t_unwind, w_unwind, u_unwind, u_unwind_scaled
            idx_unwind = trades_saved - self.n_periods_held
            u_unwind_pre = self.u_unwind_pre.iloc[idx_unwind]
            u_unwind_pre = u_unwind_pre.drop(self.cash_column_name)

            t_unwind = self.backtest_controller.results.u.index[idx_unwind]
            r_scale_unwind = self._cum_returns_to_scale_unwind(t_unwind, t)
            u_unwind_scaled = u_unwind_pre * r_scale_unwind

            u -= u_unwind_scaled
        # Unwind logic ends

        return u

    def _cum_returns_to_scale_unwind(self, t_unwind: dt.datetime, t: dt.datetime):
        df = self.actual_returns + 1
        df = df[(df.index >= t_unwind) & (df.index < t)]

        return df.cumprod().iloc[-1].drop(self.cash_column_name)

    def _save_data(self, name: str, t: dt.datetime, entry: pd.Series) -> None:
        try:
            getattr(self, name).loc[t] = entry
        except AttributeError:
            setattr(
                self,
                name,
                (pd.Series if np.isscalar(entry) else pd.DataFrame)(
                    index=[t], data=[entry]
                ),
            )
