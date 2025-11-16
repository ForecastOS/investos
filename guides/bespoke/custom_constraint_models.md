<h1>Creating Custom Constraint Models</h1>

**A quick note:** if you aren't using a convex optimization based investment strategy (like SPO), constraint models don't do anything!

## Extending BaseConstraint

The [BaseConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/base_constraint.py) class provides a foundational structure for creating custom constraint models.

Below is a step-by-step guide for extending BaseConstraint.

### Import Required Modules:

First, ensure you have the necessary modules imported:

```python
import datetime as dt
import pandas as pd
import numpy as np
import cvxpy as cvx

from investos.portfolio.constraint_model import BaseConstraint
from investos.util import get_value_at_t
```

### Define the Custom Constraint Class:

Subclass `BaseConstraint` to implement your desired constraint model.

```python
class CustomConstraint(BaseConstraint):
```

### Initialize Custom Attributes (Optional):

You may want to add additional attributes specific to your constraint model. Override the `__init__` method:

```python
def __init__(self, *args, custom_param=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.custom_param = custom_param
```

### Implement the `_weight_expr` Method:

**This is the core method** where your constraint logic resides.

Given a datetime `t`, a numpy-like array of asset holding weights `w_plus`, a numpy-like array of trade weights `z`, and a portfolio value `v`, return a `CVXPY` constraint expression.

See [MaxLeverageConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/leverage_constraint.py) for inspiration:

```python
def _weight_expr(self, t, w_plus, z, v):
    """
    Returns a series of holding constraints.

    Parameters
    ----------
    t : datetime

    w_plus : array
        Portfolio weights after trades z.

    z : array
        Trades for period t

    v : float
        Value of portfolio at period t

    Returns
    -------
    array
        The holding constraints based on the portfolio leverage after trades.
    """
    return cvx.sum(cvx.abs(w_plus)) <= self.limit

```

### Test Your Constraint Model:

You can test that your custom constraint model generates constraints as expected for a specific datetime period:

```python
actual_returns = pd.DataFrame(...)  # Add your data here. Each asset should be a column, and it should be indexed by datetime
initial_holdings = pd.Series(...)  # Holding values, indexed by asset

strategy = SPO(
    actual_returns=actual_returns,
    constraints=[CustomConstraint]
)

trade_list = strategy.generate_trade_list(
    initial_holdings,
    dt.datetime.now()
)
```

You can also plug your custom constraint model into BacktestController (through your investment strategy) to run a full backtest!

```python
backtest_controller = inv.portfolio.BacktestController(
    strategy=strategy
)
```
