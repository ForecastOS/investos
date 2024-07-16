<h1>Single Period Optimization (SPO)</h1>

## What We Need

In order to backtest a portfolio using [SPO](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/strategy/spo.py), we'll need:

-   Forecast stock returns over the time periods we wish to backtest: `forecast_returns`
-   Actual stock returns over the time periods we wish to backtest: `actual_returns`
-   Start and end dates: `start_date` and `end_date`

For [TradingCost](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/cost_model/trading_cost.py), we'll need:

-   Actual stock prices over the time periods we want to backtest: `actual_prices`
-   Forecast volume (perhaps an average of the last year for each asset, for simplicity) over the time periods we want to backtest: `forecast_volume`
-   Forecast standard deviation in returns (perhaps the standard deviation of the last year for each asset, for simplicity) over the time periods we want to backtest: `forecast_std_dev`
-   Forecast (half) trading spreads for each asset (we are using 2.5bps for all assets for simplicity in this example) over the time periods we want to backtest: `half_spread`

For [ShortHoldingCost](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/cost_model/short_holding_cost.py): we'll need:

-   Forecast short borrowing rates for each asset (we are using 40bps for all assets for simplicity in this example) over the time periods we want to backtest: `short_rates`

In order to make this example as easy as possible, we've prepared, and will use, actual returns and prices and forecast returns, volumes, standard deviations, spreads, and short rates from 2017 - 2018 for a universe of 319 stocks.

## Sample Code For a SPO Backtest

Set up modules and load data:

```python
import pandas as pd
import investos as inv
from investos.portfolio.cost_model import *
from investos.portfolio.constraint_model import *

actual_returns = pd.read_parquet("https://investos.io/example_actual_returns.parquet")
forecast_returns = pd.read_parquet("https://investos.io/example_forecast_returns.parquet")

# For trading costs:
actual_prices = pd.read_parquet("https://investos.io/example_spo_actual_prices.parquet")
forecast_volume = pd.Series(
    pd.read_csv("https://investos.io/example_spo_forecast_volume.csv", index_col="asset")
    .squeeze(),
    name="forecast_volume"
)
forecast_std_dev = pd.Series(
    pd.read_csv("https://investos.io/example_spo_forecast_std_dev.csv", index_col="asset")
    .squeeze(),
    name="forecast_std_dev"
)
half_spread_percent = 2.5 / 10_000 # 2.5 bps
half_spread = pd.Series(index=forecast_returns.columns, data=half_spread_percent)

# For short holding costs:
short_cost_percent = 40 / 10_000 # 40 bps
trading_days_per_year = 252
short_rates = pd.Series(index=forecast_returns.columns, data=short_cost_percent/trading_days_per_year)
```

Run SPO:

```python
strategy = inv.portfolio.strategy.SPO(
    actual_returns = actual_returns,
    forecast_returns = forecast_returns,
    costs = [
        ShortHoldingCost(short_rates=short_rates, exclude_assets=["cash"]),
        TradingCost(
            actual_prices=actual_prices,
            forecast_volume=forecast_volume,
            forecast_std_dev=forecast_std_dev,
            half_spread=half_spread,
            exclude_assets=["cash"],
        ),
    ],
    constraints = [
        MaxShortLeverageConstraint(limit=0.3),
        MaxLongLeverageConstraint(limit=1.3),
        MinWeightConstraint(limit=-0.03),
        MaxWeightConstraint(limit=0.03),
        LongCashConstraint(),
        MaxAbsTurnoverConstraint(limit=0.05),
    ],
    cash_column_name="cash"
)

portfolio = inv.portfolio.BacktestController(
    strategy=strategy,
    start_date='2017-01-01',
    end_date='2018-01-01',
    hooks = {
        "after_trades": [
            lambda backtest, t, u, h_next: print(".", end=''),
        ]
    }
)

backtest_result = portfolio.generate_positions()
backtest_result.summary
```

When `backtest_result.summary` is executed, it will output summary backtest results:

```python
# Initial timestamp                         2017-01-03 00:00:00
# Final timestamp                           2017-12-29 00:00:00
# Total portfolio return (%)                              15.5%
# Annualized portfolio return (%)                        15.75%
# Annualized excess portfolio return (%)                 12.68%
# Annualized excess risk (%)                              7.82%
# Information ratio (x)                                   1.62x
# Annualized risk over risk-free (%)                      7.82%
# Sharpe ratio (x)                                        1.62x
# Max drawdown (%)                                        3.54%
# Annual turnover (x)                                   457.36x
# Portfolio hit rate (%)                                  53.6%
```

What a difference trading costs make vs our previous RankLongShort example!

If you have a charting library installed, like matplotlib, check out [BaseResult](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/result/base_result.py) for the many `backtest_result` metrics you can plot!

## Next: Reporting

Next, let's explore the backtest performance reporting available to you through `backtest_result` (an instance of BaseResult): [Analyzing Backtest Results](/guides/reporting/analyzing_backtest_results).
