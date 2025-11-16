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
from investos.util import _solve_and_extract_trade_weights, get_value_at_t


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
        weights_portfolio = holdings / value  # Portfolio weights
        weights_trades = cvx.Variable(weights_portfolio.size)  # Portfolio trades
        weights_portfolio_plus_trades = (
            weights_portfolio.values + weights_trades
        )  # Portfolio weights after trades

        # JUST CURRENT TRANCHE FOR SPO_TRANCHES
        alpha_term = cvx.sum(
            cvx.multiply(
                get_value_at_t(self.forecast_returns, t).values, weights_trades
            )
        )

        assert alpha_term.is_concave()

        costs, constraints = [], []

        for cost in self.costs:
            cost_expr, const_expr = cost.cvxpy_expression(
                t, weights_trades, weights_trades, value, holdings.index
            )  # weights_trades only: only calc costs per tranche, no weights_portfolio_plus_trades for tranche opt class
            costs.append(cost_expr)
            constraints += const_expr

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

        # For help debugging:
        for el in costs:
            if not el.is_convex():
                print(t, el, "is not convex")

        for el in constraints:
            if not el.is_dcp():
                print(t, el, "is not dcp")

        objective = cvx.Maximize(alpha_term - cvx.sum(costs))
        constraints += [cvx.sum(weights_trades) == 0]
        prob = cvx.Problem(
            objective, constraints
        )  # Trades need to 0 out, i.e. cash account must adjust to make everything net to 0

        return (prob, weights_trades)

    def precompute_trades_distributed(self, holdings: pd.Series, time_periods):
        delayed_tasks = []

        for t in time_periods:
            prob, weights_trades = self.formulate_optimization_problem(holdings, t)
            delayed_task = delayed(_solve_and_extract_trade_weights)(
                prob, weights_trades, t, self.solver, self.solver_opts, holdings
            )
            delayed_tasks.append(delayed_task)

        print(f"\nComputing trades at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
        results = compute(*delayed_tasks)
        print(
            f"\nFinished computing trades at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
        )

        print("\nSaving distributed trades...")
        for t, weights_trades in results:
            self._save_data("weights_trades_distr", t, weights_trades)
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
            weights_trades = self.weights_trades_distr.loc[t].values

        else:
            prob, weights_trades = self.formulate_optimization_problem(holdings, t)

            try:
                prob.solve(solver=self.solver, **self.solver_opts)

                if prob.status in ("unbounded", "infeasible"):
                    print(f"The problem is {prob.status} at {t}.")
                    return self._zerotrade(holdings)

                weights_trades = weights_trades.value

            except (cvx.SolverError, cvx.DCPError, TypeError) as e:
                print(f"The solver failed for {t}. Error details: {e}")
                return self._zerotrade(holdings)

        value = sum(holdings)

        try:
            dollars_trades = pd.Series(
                index=holdings.index, data=(weights_trades * value)
            )
        except Exception as e:
            print(f"Calculating trades failed for {t}. Error details: {e}")
            return self._zerotrade(holdings)

        # Zero out small values; cash (re)calculated later based on trade balance, cash value here doesn't matter
        if self.polishing:
            dollars_trades[abs(dollars_trades) < value / self.polishing_denom] = 0

        # Round trade to discreet n_share_block (default: 100)
        if self.discreet_shares:
            prices = get_value_at_t(self.actual_prices, t)
            block_prices = prices * self.n_share_block
            block_prices[self.cash_column_name] = None

            non_cash_mask = dollars_trades.index != self.cash_column_name
            dollars_trades[non_cash_mask] = (
                dollars_trades[non_cash_mask] / block_prices[non_cash_mask]
            ).round() * block_prices[non_cash_mask]

        # Unwind logic starts
        trades_saved = self.backtest_controller.results.dollars_trades.shape[0]
        self._save_data("dollars_trades_unwind_pre", t, dollars_trades)

        if trades_saved > self.n_periods_held:
            idx_unwind = trades_saved - self.n_periods_held
            t_unwind = self.backtest_controller.results.dollars_trades.index[idx_unwind]

            dollars_trades_unwind_pre = self.dollars_trades_unwind_pre.loc[t_unwind]
            dollars_trades_unwind_pre = dollars_trades_unwind_pre.drop(
                self.cash_column_name
            )

            r_scale_unwind = self._cum_returns_to_scale_unwind(t_unwind, t)
            dollars_trades_unwind_scaled = dollars_trades_unwind_pre * r_scale_unwind

            dollars_trades -= dollars_trades_unwind_scaled
        # Unwind logic ends

        return dollars_trades

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
