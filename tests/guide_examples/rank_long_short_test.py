import pandas as pd

import investos as inv


def test_rank_long_short():
    actual_returns = pd.read_parquet(
        "https://investos.io/example_actual_returns.parquet"
    )
    forecast_returns = pd.read_parquet(
        "https://investos.io/example_forecast_returns.parquet"
    )

    strategy = inv.portfolio.strategy.RankLongShort(
        actual_returns=actual_returns,
        metric_to_rank=forecast_returns,
        leverage=1.6,
        ratio_long=130,
        ratio_short=30,
        percent_long=0.2,
        percent_short=0.2,
        n_periods_held=60,
        cash_column_name="cash",
    )

    portfolio = inv.portfolio.BacktestController(
        strategy=strategy,
        start_date="2017-01-01",
        end_date="2018-01-01",
        aum=100_000_000,
    )

    backtest_result = portfolio.generate_positions()
    summary = backtest_result._summary_string()

    assert isinstance(summary, str)
    assert round(backtest_result.annualized_return, 4) == 0.1749
    assert round(backtest_result.excess_risk_annualized, 4) == 0.0609
    assert round(backtest_result.information_ratio, 2) == 2.37
    assert round(backtest_result.annual_turnover, 2) == 9.97
    assert round(backtest_result.portfolio_hit_rate, 3) == 0.6000
