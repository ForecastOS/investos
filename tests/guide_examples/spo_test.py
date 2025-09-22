import pandas as pd

import investos as inv
from investos.portfolio.constraint_model import (
    LongCashConstraint,
    MaxAbsTurnoverConstraint,
    MaxLongLeverageConstraint,
    MaxShortLeverageConstraint,
    MaxWeightConstraint,
    MinWeightConstraint,
)
from investos.portfolio.cost_model import ShortHoldingCost, TradingCost


def test_spo():
    actual_returns = pd.read_parquet(
        "https://investos.io/example_actual_returns.parquet"
    )
    forecast_returns = pd.read_parquet(
        "https://investos.io/example_forecast_returns.parquet"
    )

    # For trading costs:
    actual_prices = pd.read_parquet(
        "https://investos.io/example_spo_actual_prices.parquet"
    )
    forecast_volume = pd.Series(
        pd.read_csv(
            "https://investos.io/example_spo_forecast_volume.csv", index_col="asset"
        ).squeeze(),
        name="forecast_volume",
    )
    forecast_std_dev = pd.Series(
        pd.read_csv(
            "https://investos.io/example_spo_forecast_std_dev.csv", index_col="asset"
        ).squeeze(),
        name="forecast_std_dev",
    )
    half_spread_percent = 2.5 / 10_000  # 2.5 bps
    half_spread = pd.Series(index=forecast_returns.columns, data=half_spread_percent)

    # For short holding costs:
    short_cost_percent = 40 / 10_000  # 40 bps
    trading_days_per_year = 252
    short_rates = pd.Series(
        index=forecast_returns.columns, data=short_cost_percent / trading_days_per_year
    )

    strategy = inv.portfolio.strategy.SPO(
        actual_returns=actual_returns,
        forecast_returns=forecast_returns,
        costs=[
            ShortHoldingCost(short_rates=short_rates, exclude_assets=["cash"]),
            TradingCost(
                actual_prices=actual_prices,
                forecast_volume=forecast_volume,
                forecast_std_dev=forecast_std_dev,
                half_spread=half_spread,
                exclude_assets=["cash"],
            ),
        ],
        constraints=[
            MaxShortLeverageConstraint(limit=0.3),
            MaxLongLeverageConstraint(limit=1.3),
            MinWeightConstraint(limit=-0.03),
            MaxWeightConstraint(limit=0.03),
            LongCashConstraint(),
            MaxAbsTurnoverConstraint(limit=0.05),
        ],
        cash_column_name="cash",
        solver_opts={
            "eps_abs": 1e-5,
            "eps_rel": 1e-5,
            "adaptive_rho_interval": 50,
        },
    )

    portfolio = inv.portfolio.BacktestController(
        strategy=strategy,
        start_date="2017-01-01",
        end_date="2018-01-01",
        hooks={
            "after_trades": [
                lambda backtest, t, u, h_next: print(".", end=""),
            ]
        },
    )

    backtest_result = portfolio.generate_positions()
    summary = backtest_result._summary_string()

    print(summary)

    assert isinstance(summary, str)
    assert (
        round(backtest_result.annualized_return, 2) >= 0.05
        and round(backtest_result.annualized_return, 2) <= 0.06
    )
    assert (
        round(backtest_result.annual_turnover, 1) >= 12.6
        and round(backtest_result.annual_turnover, 1) <= 13.0
    )
    assert (
        round(backtest_result.portfolio_hit_rate, 2) >= 0.56
        and round(backtest_result.portfolio_hit_rate, 2) <= 0.60
    )
