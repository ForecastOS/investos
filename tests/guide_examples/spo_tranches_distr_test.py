import pandas as pd

import investos as inv
from investos.portfolio.constraint_model import (
    LongCashConstraint,
    MaxLongTradeLeverageConstraint,
    MaxShortTradeLeverageConstraint,
    MaxTradeWeightConstraint,
    MinTradeWeightConstraint,
)
from investos.portfolio.cost_model import ShortHoldingCost, TradingCost


def test_spo_tranches():
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

    n_periods_held = 30

    strategy = inv.portfolio.strategy.SPOTranches(
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
            MaxLongTradeLeverageConstraint(limit=1.3 / n_periods_held),
            MaxShortTradeLeverageConstraint(limit=0.3 / n_periods_held),
            MinTradeWeightConstraint(limit=-0.03 / n_periods_held),
            MaxTradeWeightConstraint(limit=0.03 / n_periods_held),
            LongCashConstraint(),
        ],
        n_periods_held=n_periods_held,
        cash_column_name="cash",
        solver_opts={
            "eps_abs": 1e-6,
            "eps_rel": 1e-6,
            "adaptive_rho_interval": 50,
        },
    )

    portfolio = inv.portfolio.BacktestController(
        strategy=strategy,
        distributed=True,
        dask_cluster_config={
            "n_workers": 10,
            "environment": {
                "EXTRA_PIP_PACKAGES": "investos scikit-learn numpy>=2.0",
            },
        },
        start_date="2017-01-01",
        end_date="2017-06-30",
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
        round(backtest_result.annualized_return, 3) >= 0.033
        and round(backtest_result.annualized_return, 3) <= 0.037
    )
    assert (
        round(backtest_result.annual_turnover, 1) >= 8.3
        and round(backtest_result.annual_turnover, 1) <= 9.3
    )
    assert (
        round(backtest_result.portfolio_hit_rate, 2) >= 0.67
        and round(backtest_result.portfolio_hit_rate, 2) <= 0.77
    )
