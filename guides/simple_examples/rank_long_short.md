<h1>Rank Long Short</h1>

## What We Need

In order to backtest a portfolio using [RankLongShort](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/strategy/rank_long_short.py), we'll need:

-   A metric to rank assets by over time: `metric_to_rank`
    -   In this example, we'll use forecast returns for stocks, but we could also use LTM sales, age of CEO, etc.
-   Stock returns over the time periods we wish to backtest: `actual_returns`
-   Start and end dates: `start_date` and `end_date`

In order to make this example as easy as possible, we've prepared, and will use, forecast and actual returns from 2017 - 2018 for a universe of 319 stocks. We will also exclude cost models.

## Sample Code For a RankLongShort Backtest

```python
import pandas as pd
import investos as inv

actual_returns = pd.read_parquet("https://investos.io/example_actual_returns.parquet")
forecast_returns = pd.read_parquet("https://investos.io/example_forecast_returns.parquet")

strategy = inv.portfolio.strategy.RankLongShort(
    actual_returns = actual_returns,
    metric_to_rank = forecast_returns,
    leverage=1.6,
    ratio_long=130,
    ratio_short=30,
    percent_long=0.2,
    percent_short=0.2,
    n_periods_held=60,
    cash_column_name="cash"
)

portfolio = inv.portfolio.BacktestController(
    strategy=strategy,
    start_date='2017-01-01',
    end_date='2018-01-01',
    aum=100_000_000
)

backtest_result = portfolio.generate_positions()
backtest_result.summary
```

That's all that's required to run your first (RankLongShort) backtest!

When `backtest_result.summary` is executed, it will output summary backtest results:

```python
# Initial timestamp                         2017-01-03 00:00:00
# Final timestamp                           2017-12-29 00:00:00
# Total portfolio return (%)                             17.22%
# Annualized portfolio return (%)                        17.49%
# Annualized excess portfolio return (%)                 14.42%
# Annualized excess risk (%)                              6.09%
# Information ratio (x)                                   2.37x
# Annualized risk over risk-free (%)                      6.09%
# Sharpe ratio (x)                                        2.37x
# Max drawdown (%)                                        3.21%
# Annual turnover (x)                                     9.97x
# Portfolio hit rate (%)                                  60.0%
```

If you have a charting library installed, like matplotlib, check out [BaseResult](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/result/base_result.py) for the many metrics you can plot, like:

-   portfolio value evolution (`backtest_result.v`),
-   long and short leverage evolution (`backtest_result.leverage`),
-   trades in SBUX (`backtest_result.trades['SBUX']`),
-   holdings in AAPL (`backtest_result.h['AAPL']`),
-   etc.

## What Could Be Improved

In the above example, for simplicity, we:

-   Didn't use any cost models
    -   e.g. TradingCost, ShortHoldingCost
-   Assumed our initial portfolio was all cash
    -   You can override this by setting the `initial_portfolio` kwarg equal to a (Pandas) series of asset values when initializing [BacktestController](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/backtest_controller.py#L19).

## Next: Single Period Optimization

Next, let's explore adding cost and constraint models in our next guide: [Single Period Optimization](/guides/simple_examples/spo).
