<h1>Rank Long Short</h1>

## What We Need

In order to backtest a portfolio using [RankLongShort](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/strategy/rank_long_short.py), we'll need:

-   A metric to rank assets by over time: `metric_to_rank`
    -   In this example, we'll use forecast returns for stocks, but we could also use LTM sales, age of CEO, etc.
-   Stock returns over the time periods we wish to backtest: `actual_returns`
-   Start and end dates: `start_date` and `end_date`

In order to make this example as easy as possible, we've prepared, and will use, forecast and actual returns from 2017 - 2018 for a universe of 319 stocks.

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
# Total portfolio return (%)                             52.06%
# Annualized portfolio return (%)                        52.99%
# Annualized excess portfolio return (%)                  49.9%
# Annualized excess risk (%)                               8.0%
# Information ratio (x)                                   6.24x
# Annualized risk over risk-free (%)                       8.0%
# Sharpe ratio (x)                                        6.24x
# Max drawdown (%)                                        2.11%
# Annual turnover (x)                                   609.25x
# Portfolio hit rate (%)                                  61.2%
```

If you have a charting library installed, like matplotlib, check out [BaseResult](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/result/base_result.py) for the many metrics you can plot, like portfolio value (`backtest_result.v`):

<img src="/guide_images/rank_long_short_v_evo.png" alt="Portfolio value evolution for RankLongShort" width="400" />

long and short leverage (`backtest_result.leverage`):

<img src="/guide_images/rank_long_short_lev_evo.png" alt="Portfolio leverage evolution for RankLongShort" width="400" />

trades in SBUX (`backtest_result.trades['SBUX']`) or holdings in AAPL (`backtest_result.h['AAPL']`):

<img src="/guide_images/rank_long_short_h_aapl_evo.png" alt="Portfolio leverage evolution for RankLongShort" width="400" />

etc.

## What Could Be Improved

In the above example, for simplicity, we:

-   Didn't use any cost models
    -   e.g. TradingCost, ShortHoldingCost
-   Assumed our initial portfolio was all cash
    -   You can override this by setting the `initial_portfolio` kwarg equal to a (Pandas) series of asset values when initializing [BacktestController](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/backtest_controller.py#L19).

## Next: Single Period Optimization

Next, let's explore adding cost and constraint models in our next guide: [Single Period Optimization](/guides/simple_examples/spo).
