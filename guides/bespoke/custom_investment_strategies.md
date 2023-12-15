<h1>Creating Custom Investment Strategies</h1>

## Extending BaseStrategy

The [BaseStrategy](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/strategy/base_strategy.py) class provides a foundational structure for creating custom investment strategies.

Below is a step-by-step guide for extending BaseStrategy.

### Import Required Modules:

First, ensure you have the necessary modules imported:

```python
import datetime as dt
import pandas as pd
from investos.portfolio.strategy import BaseStrategy
from investos.util import values_in_time
```

### Define the Custom Strategy Class:

Subclass `BaseStrategy` to implement the desired strategy.

```python
class CustomStrategy(BaseStrategy):
```

### Initialize Custom Attributes (Optional):

You may want to add additional attributes specific to your strategy. Override the `__init__` method:

```python
def __init__(self, *args, custom_param=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.custom_param = custom_param
```

### Implement the `generate_trade_list` Method:

**This is the core method** where your strategy logic resides.

Given a series of holdings indexed by asset, and a date `t`, it should calculate and return a trade list series, also indexed by asset.

For example, a simple, contrived momentum-based strategy might look like this:

```python
def generate_trade_list(self, holdings: pd.Series, t: dt.datetime) -> pd.Series:
    # A placeholder example:
    ### Buy assets that have had positive returns in the last period
    returns = values_in_time(self.actual_returns, t)
    buy_assets = returns[returns > 0].index
    trade_values = pd.Series(index=holdings.index, data=0.0)
    trade_values[buy_assets] = 100  # Buying $100 of each positive-return asset

    return trade_values
```

### Implement Helper Methods (Optional):

You can add custom helper methods to factor in specific logic or utilities that help in constructing your strategy (and help in keeping your logic understandable).

### Test Your Strategy:

You can test that your custom strategy generates trades as expected for a specific datetime period:

```python
actual_returns = pd.DataFrame(...)  # Add your data here. Each asset should be a column, and it should be indexed by datetime
initial_holdings = pd.Series(...)  # Holding values, indexed by asset

strategy = CustomStrategy(
    actual_returns=actual_returns,
    custom_param="example_value"
)

trade_list = strategy.generate_trade_list(
    initial_holdings,
    dt.datetime.now()
)

print(trade_list)
```

You can also plug your custom strategy into BacktestController to run a full backtest!

```python
backtest_controller = inv.portfolio.BacktestController(
    strategy=strategy
)
```

### Integrate Costs and Constraints:

Use the `costs` parameter (in `BaseStrategy`) to incorporate costs.

If your strategy uses convex optimization, use the `constraints` parameter (in `BaseStrategy`) to incorporate constraints.

---

## Customizing Existing Strategies

Both [RankLongShort](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/strategy/rank_long_short.py) and [SPO](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/strategy/spo.py) classes, which extend [BaseStrategy](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/strategy/base_strategy.py), offer a base foundation for implementing investment strategies.

While their structure is designed for general use, for users seeking to implement more advanced and nuanced strategies, their architecture supports customization and extension.

### General Extension Ideas

1. **Dynamic Leverage**: Rather than using a fixed leverage ratio, you could dynamically adjust leverage based on market volatility, market sentiment, or other indicators.

2. **Adaptive Percent Long/Short**: Adjust the percentage of assets that are long or short based on changing market conditions. For example, during bullish markets, increase the percent long.

### RankLongShort Extension Ideas

1. **Sector-Based Ranking**: Instead of ranking all assets, classify them by sectors and rank within each sector. This can ensure diversification across various sectors.

2. **Custom Weighting Mechanisms**: Override the `_get_trade_weights_for_t` method to use custom weights beyond simple ranking, perhaps factoring in other metrics such as asset volatility or liquidity.

3. **Custom Unwind Logic**: Modify the unwinding logic for positions to not just be based on time but on market conditions, metrics, or other constraints.

### SPO Extension Ideas

1. **Post-Solution Adjustments**: After the solver has provided a solution, make post-optimization adjustments, perhaps for real-world considerations like rounding off to whole shares or accounting for latest market prices.

2. **Advanced Error Handling**: Instead of just handling errors by zeroing trades when the solver fails, consider implementing a fallback mechanism or use an alternative optimization approach.

### Final Thoughts

-   Like BaseStrategy, both RankLongShort and SPO are designed to be extensible.

-   Make sure you test extensively with historical data before deploying to a live environment!
