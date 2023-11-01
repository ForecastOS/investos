<h1>Single Period Optimization (SPO)</h1>

## What We Need

In order to backtest a portfolio using [SPO](https://github.com/ForecastOS/investos/blob/1d5fb91ab2e36f2014b5b26fe0e6001f5b89321d/investos/portfolio/strategy/spo.py), we'll need:

-   Forecast stock returns over the time periods we wish to backtest: `forecast_returns`
-   Actual stock returns over the time periods we wish to backtest: `actual_returns`
-   Start and end dates: `start_date` and `end_date`

In order to make this example as easy as possible, we've prepared, and will use, forecast and actual returns from 2017 - 2018 for a universe of 319 stocks.

## Sample Code For a SPO Backtest

```python
import pandas as pd
import investos as inv

actual_returns = pd.read_parquet("https://investos.io/example_actual_returns.parquet")
forecast_returns = pd.read_parquet("https://investos.io/example_forecast_returns.parquet")

strategy = inv.portfolio.strategy.SPO(
    actual_returns = actual_returns,
    metric_to_rank = forecast_returns,
    cash_column_name="cash"
)

portfolio = inv.portfolio.BacktestController(
    strategy=strategy,
    start_date='2017-01-01',
    end_date='2018-01-01',
    aum=100_000_000
)

backtest_result.summary
```
