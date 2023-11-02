<h1>Creating Custom Cost Models</h1>

## Extending BaseCost

The [BaseCost](https://github.com/ForecastOS/investos/blob/1d5fb91ab2e36f2014b5b26fe0e6001f5b89321d/investos/portfolio/cost_model/base_cost.py) class provides a foundational structure for creating custom cost models.

Below is a step-by-step guide for extending BaseCost.

### Import Required Modules:

First, ensure you have the necessary modules imported:

```python
import datetime as dt
import pandas as pd
import numpy as np
from investos.portfolio.cost_model import BaseCost
from investos.util import values_in_time
```

### Define the Custom Cost Class:

Subclass `BaseCost` to implement your desired cost model.

```python
class CustomCost(BaseCost):
```

### Initialize Custom Attributes (Optional):

You may want to add additional attributes specific to your cost model. Override the `__init__` method:

```python
def __init__(self, *args, custom_param=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.custom_param = custom_param
```

### Implement the `get_actual_cost` Method:

**This is the core method** where your cost logic resides.

Given a datetime `t`, a series of holdings indexed by asset `h_plus`, and a series of trades indexed by asset `u`, return the sum of costs for all assets.

See [ShortHoldingCost](https://github.com/ForecastOS/investos/blob/1d5fb91ab2e36f2014b5b26fe0e6001f5b89321d/investos/portfolio/cost_model/short_holding_cost.py) for inspiration:

```python
def get_actual_cost(
        self, t: dt.datetime, h_plus: pd.Series, u: pd.Series
    ) -> pd.Series:
    """Method that calculates per-period (short position) holding costs given period `t` holdings and trades.

    Parameters
    ----------
    t : datetime.datetime
        The datetime for associated trades `u` and holdings plus trades `h_plus`.
    h_plus : pandas.Series
        Holdings at beginning of period t, plus trades for period `t` (`u`). Same as `u` + `h` for `t`.
    u : pandas.Series
        Trades (as values) for period `t`.
    """
    return sum(-np.minimum(0, h_plus) * self._get_short_rate(t))

def _get_short_rate(self, t):
    return values_in_time(self.short_rates, t)
```

### Implement the `_estimated_cost_for_optimization` Method (Optional):

If you're using a convex optimization based investment strategy, `_estimated_cost_for_optimization` is used to return a cost expression for optimization.

Given a datetime `t`, a numpy-like array of holding weights `w_plus`, and a numpy-like array of trade weights `z`, return a two item tuple containing a `cvx.sum(expression)` and a (possibly empty) list of constraints.

See [ShortHoldingCost](https://github.com/ForecastOS/investos/blob/1d5fb91ab2e36f2014b5b26fe0e6001f5b89321d/investos/portfolio/cost_model/short_holding_cost.py) for inspiration:

```python
def _estimated_cost_for_optimization(self, t, w_plus, z, value):
    """Estimated holding costs.

    Used by optimization strategy to determine trades.

    Not used to calculate simulated holding costs for backtest performance.
    """
    expression = cvx.multiply(self._get_short_rate(t), cvx.neg(w_plus))

    return cvx.sum(expression), []
```

### Implement Helper Methods (Optional):

You can add custom helper methods to factor in specific logic or utilities that help in constructing your cost model (and help in keeping your logic understandable).

### Test Your Cost Model:

You can test that your custom cost model generates costs as expected for a specific datetime period:

```python
actual_returns = pd.DataFrame(...)  # Add your data here. Each asset should be a column, and it should be indexed by datetime
initial_holdings = pd.Series(...)  # Holding values, indexed by asset

strategy = SPO(
    actual_returns=actual_returns,
    costs=[CustomCost]
)

trade_list = strategy.generate_trade_list(
    initial_holdings,
    dt.datetime.now()
)
```

You can also plug your custom cost model into BacktestController (through your investment strategy) to run a full backtest!

```python
backtest_controller = inv.portfolio.BacktestController(
    strategy=strategy
)
```
