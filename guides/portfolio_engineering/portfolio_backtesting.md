<h1>Portfolio Backtesting</h1>

This guide covers:

1.  Backtesting your portfolio with InvestOS
2.  Viewing backtest results

## Backtesting your portfolio

There are generally two types of InvestOS users:

1. Users that create their portfolio _using InvestOS_
    - i.e. an InvestOS optimized portfolio
2. Users that create their portfolio _outside of InvestOS_
    - i.e. a time-series asset weight portfolio

We've created examples for both use-cases below.

### InvestOS optimized portfolio

If your portfolio was created / optimized using InvestOS, a backtest was automatically run on your behalf. You can access that backtest result as follows:

```python
# 1. Initializing the optimization strategy
# -- and controller for your portfolio

strategy = inv.portfolio.strategy.SPO(**optional_args)

portfolio = inv.portfolio.Controller(
    df_forecast,
    df_actual,
    strategy=strategy,
    **optional_args
)

# 2. Running the optimization and generating your positions,
# -- returning a BaseResult
# -- (investos/portfolio/result/base_result)
# -- instance with backtest results.

result = portfolio.generate_positions()
```

That's all that's required; the `result` object contains your backtest.

### Time-series asset weight portfolio

Backtesting a portfolio created outside of InvestOS requires several (fairly simple) initial steps.

They are detailed below.

#### Generate / compile required data

Let's assume we have a time-series asset weight portfolio in the following format, saved as a CSV:

```
| date               | asset        | weight |
| ------------------ | ------------ | ------ |
| 2021-10-12 9:30:00 | AAPL         | 0.10   |
| 2021-10-12 9:30:00 | MSFT         | 0.15   |
| 2021-10-12 9:30:00 | GOOGL        | 0.12   |
| 2021-10-12 9:30:00 | AMZN         | 0.18   |
| 2021-10-12 9:30:00 | TSLA         | 0.20   |
| 2021-10-12 9:30:00 | FB           | 0.08   |
| 2021-10-12 9:30:00 | NFLX         | 0.05   |
| 2021-10-12 9:30:00 | TWTR         | 0.07   |
| 2021-10-12 9:30:00 | NVDA         | 0.04   |
| 2021-10-12 9:30:00 | AMD          | 0.01   |
| ...                | ...          | ...    |
```

Let's also assume you have a time-series asset price history in the following format, saved as a CSV:

```
| date               | asset        | price  |
| ------------------ | ------------ | ------ |
| 2021-10-12 9:30:00 | AAPL         | 1.01   |
| 2021-10-12 9:30:00 | MSFT         | 2.02   |
| 2021-10-12 9:30:00 | GOOGL        | 3.03   |
| 2021-10-12 9:30:00 | AMZN         | 4.04   |
| 2021-10-12 9:30:00 | TSLA         | 5.05   |
| 2021-10-12 9:30:00 | FB           | 6.06   |
| 2021-10-12 9:30:00 | NFLX         | 7.07   |
| 2021-10-12 9:30:00 | TWTR         | 8.08   |
| 2021-10-12 9:30:00 | NVDA         | 9.09   |
| 2021-10-12 9:30:00 | AMD          | 1.11   |
| ...                | ...          | ...    |
```

#### Load asset weights

```python
import pandas as pd
import numpy as np

df_weights = pd.read_csv('weights.csv')
df_weights = df_weights.pivot(index="date", columns="asset", values="weight")
```

#### Calculate returns

```python
df_returns = pd.read_csv('returns.csv')
df_returns['return'] = df_returns.groupby('asset')['price'].pct_change()
df_returns = df_returns.pivot(index="date", columns="asset", values="return")

# Convert to fwd period return
df_returns = df_returns.shift(-1)
```

#### Scale weights for returns

```python
s_scale_weights = (
    df_weights.shift(1).fillna(0) * df_returns.shift(1).fillna(0)
).sum(axis=1) + 1
s_scale_weights = s_scale_weights.cumprod()
df_weights = df_weights.multiply(s_scale_weights, axis=0)
```

#### Infer trades

```python
# Start weights at 0 for T=0 (before AUM is invested)
new_row = pd.DataFrame(
    [], index=[df_weights.index[0] - pd.Timedelta(days=1)]
)

df_returns = pd.concat([new_row, df_returns]).fillna(0)
df_weights = pd.concat([new_row, df_weights]).fillna(0)

# Infer trade weights
df_trades = df_weights - df_weights.shift(1) * (1 + df_returns.shift(1))
df_trades = df_trades.fillna(0)
```

#### Run backtest for weights

```python
import investos as inv
from investos.portfolio.result import WeightsResult

# Calculate cash trades, add cash returns
df_weights["cash"] = 1
df_trades["cash"] = 0
df_trades["cash"] -= df_trades.sum(axis=1)
# df_returns["cash"] series could be
# -- the 90 day T-bill yield, a constant number,
# -- 0 (for simplicity), etc.
df_returns["cash"] = 0.04 / 252 # (4% / 252 trading days)

result = WeightsResult(
    initial_weights=df_weights.iloc[0],
    trade_weights=df_trades,
    returns=df_returns,
    risk_free=df_returns["cash"], # Add any series you want
    benchmark=df_returns["cash"], # Add any series you want
    aum=100_000_000,
)
```

#### Result object

Getting the `result` backtest object takes a few more steps for portfolios created outside of InvestOS, but you end up with the same `result` object.

Let's explore backtest results next.

## Exploring backtest results

You can print summary results for you backtest with the following code:

```python
result.summary
```

In an ipynb, you can easily plot:

-   Your portfolio value evolution `result.v.plot()`
-   Your leverage `result.long_leverage.plot()`
-   Your holdings in a specific asset `result.h['TSLA'].plot()`
-   Your trades in a specific asset `result.trades['TSLA'].plot()`
-   And many more metrics, series, and dataframes provided by the result object

To view all of the reporting functionality available to you, [check out the result class on Github](https://github.com/ForecastOS/investos/blob/v0.2.2/investos/portfolio/result/base_result.py).

Looking for more reporting? Consider extending the [BaseResult](https://github.com/ForecastOS/investos/blob/v0.2.2/investos/portfolio/result/base_result.py) class, or opening a pull request for [InvestOS on GitHub](https://github.com/ForecastOS/investos)!

## Next: guides under development

That's it for now, but if you want to see what guide is next up, check out: [guides under development](/guides/coming_soon/guides_under_development).
