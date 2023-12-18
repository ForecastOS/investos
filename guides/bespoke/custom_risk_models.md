<h1>Creating Custom Risk Models</h1>

## Extending BaseRisk

The [BaseRisk](https://github.com/ForecastOS/investos/tree/v0.3.10/investos/portfolio/risk_model/base_risk.py) class provides a foundational structure for creating custom risk models.

Below is a step-by-step guide for extending BaseRisk.

### Import Required Modules:

First, ensure you have the necessary modules imported:

```python
import cvxpy as cvx
import numpy as np

import investos.util as util
from investos.portfolio.risk_model import BaseRisk
```

### Define the Custom Risk Class:

Subclass `BaseRisk` to implement your desired risk model.

```python
class CustomRisk(BaseRisk):
```

### Initialize Custom Attributes (Optional):

You may want to add additional attributes specific to your risk model. Override the `__init__` method:

```python
def __init__(self, *args, custom_param=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.custom_param = custom_param
```

### Implement the `_estimated_cost_for_optimization` Method:

`_estimated_cost_for_optimization` returns a utility cost expression for optimization that penalizes risk.

Given a datetime `t`, a numpy-like array of holding weights `w_plus`, a numpy-like array of trade weights `z`, and portfolio value `value`, return a two item tuple containing a CVXPY expression and a (possibly empty) list of constraints.

See [FactorRisk](https://github.com/ForecastOS/investos/tree/v0.3.10/investos/portfolio/risk_model/factor_risk.py) for inspiration:

```python
def _estimated_cost_for_optimization(self, t, w_plus, z, value):
        """Optimization (non-cash) cost penalty for assuming associated asset risk.

        Used by optimization strategy to determine trades.
        """
        factor_covar = util.values_in_time(
            self.factor_covariance, t, lookback_for_closest=True
        )
        factor_load = util.values_in_time(
            self.factor_loadings, t, lookback_for_closest=True
        )
        idiosync_var = util.values_in_time(
            self.idiosyncratic_variance, t, lookback_for_closest=True
        )

        self.expression = cvx.sum_squares(cvx.multiply(np.sqrt(idiosync_var), w_plus))

        risk_from_factors = factor_load.T @ factor_covar @ factor_load

        self.expression += w_plus @ risk_from_factors @ w_plus.T

        return self.expression, []
```

### Implement Helper Methods (Optional):

You can add custom helper methods to factor in specific logic or utilities that help in constructing your risk model (and help in keeping your logic understandable).

### Test Your Risk Model:

You can test that your custom risk model generates a utility penalty for risk as expected for a specific datetime period:

```python
actual_returns = pd.DataFrame(...)  # Add your data here. Each asset should be a column, and it should be indexed by datetime
initial_holdings = pd.Series(...)  # Holding values, indexed by asset

strategy = SPO(
    actual_returns=actual_returns,
    ...
    risk_model=CustomRisk(),
)

trade_list = strategy.generate_trade_list(
    initial_holdings,
    dt.datetime.now()
)
```

You can also plug your custom risk model into BacktestController (through your investment strategy) to run a full backtest!

```python
backtest_controller = inv.portfolio.BacktestController(
    strategy=strategy
)
```
