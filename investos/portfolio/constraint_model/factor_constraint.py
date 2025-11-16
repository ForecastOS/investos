import cvxpy as cvx

from investos.portfolio.constraint_model.base_constraint import BaseConstraint
from investos.util import get_value_at_t


class ZeroFactorExposureConstraint(BaseConstraint):
    def __init__(self, factor_exposure, **kwargs):
        self.factor_exposure = factor_exposure
        super().__init__(**kwargs)

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return (
            cvx.sum(
                cvx.multiply(
                    get_value_at_t(self.factor_exposure, t),
                    weights_portfolio_plus_trades,
                )
            )
            == 0
        )


class ZeroTradeFactorExposureConstraint(BaseConstraint):
    def __init__(self, factor_exposure, **kwargs):
        self.factor_exposure = factor_exposure
        super().__init__(**kwargs)

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return (
            cvx.sum(
                cvx.multiply(get_value_at_t(self.factor_exposure, t), weights_trades)
            )
            == 0
        )


class MaxFactorExposureConstraint(BaseConstraint):
    def __init__(self, factor_exposure, limit=0.05, **kwargs):
        self.factor_exposure = factor_exposure
        self.limit = limit
        super().__init__(**kwargs)

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return (
            cvx.sum(
                cvx.multiply(
                    get_value_at_t(self.factor_exposure, t),
                    weights_portfolio_plus_trades,
                )
            )
            <= self.limit
        )


class MinFactorExposureConstraint(BaseConstraint):
    def __init__(self, factor_exposure, limit=-0.05, **kwargs):
        self.factor_exposure = factor_exposure
        self.limit = limit
        super().__init__(**kwargs)

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return (
            cvx.sum(
                cvx.multiply(
                    get_value_at_t(self.factor_exposure, t),
                    weights_portfolio_plus_trades,
                )
            )
            >= self.limit
        )


class MaxAbsoluteFactorExposureConstraint(BaseConstraint):
    def __init__(self, factor_exposure, limit=0.4, **kwargs):
        self.factor_exposure = factor_exposure
        self.limit = limit
        super().__init__(**kwargs)

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return (
            cvx.sum(
                cvx.multiply(
                    get_value_at_t(self.factor_exposure, t),
                    cvx.abs(weights_portfolio_plus_trades),
                )
            )
            <= self.limit
        )


class MaxAbsoluteTradeFactorExposureConstraint(BaseConstraint):
    def __init__(self, factor_exposure, limit=0.01, **kwargs):
        self.factor_exposure = factor_exposure
        self.limit = limit
        super().__init__(**kwargs)

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return (
            cvx.sum(
                cvx.multiply(
                    get_value_at_t(self.factor_exposure, t), cvx.abs(weights_trades)
                )
            )
            <= self.limit
        )


class MaxTradeFactorExposureConstraint(BaseConstraint):
    def __init__(self, factor_exposure, limit=0.05, **kwargs):
        self.factor_exposure = factor_exposure
        self.limit = limit
        super().__init__(**kwargs)

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return (
            cvx.sum(
                cvx.multiply(get_value_at_t(self.factor_exposure, t), weights_trades)
            )
            <= self.limit
        )


class MinTradeFactorExposureConstraint(BaseConstraint):
    def __init__(self, factor_exposure, limit=-0.05, **kwargs):
        self.factor_exposure = factor_exposure
        self.limit = limit
        super().__init__(**kwargs)

    def _cvxpy_expression(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        return (
            cvx.sum(
                cvx.multiply(get_value_at_t(self.factor_exposure, t), weights_trades)
            )
            >= self.limit
        )
