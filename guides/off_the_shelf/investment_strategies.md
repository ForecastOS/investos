<h1>Using Off-The-Shelf Investment Strategies</h1>

InvestOS provides the following optimization strategies:

-   [RankLongShort](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/strategy/rank_long_short.py)
    -   Builds trade lists based on long and short positions ranked by any (possibly forecasted) metric
-   [SPO](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/strategy/spo.py) (Single Period Optimization)
    -   Builds trade lists using single period convex optimization
    -   Uses [CVXPY](https://www.cvxpy.org/tutorial/intro/index.html)
    - **Optimization**: maximizes expected return, less costs, while adhering to constraints
-   [SPOTranches](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/strategy/spo_tranches.py) (Single Period Optimization Tranches)
    -   Like SPO, but builds portfolio in separate, optimized tranches. Tranches are cycled in and out by customizable holding period. Tranches can be analyzed and altered in flight using BacktestController hooks
    -   Uses [CVXPY](https://www.cvxpy.org/tutorial/intro/index.html)
    - **Optimization**: maximizes expected return, less costs, while adhering to constraints
-   [SPOWeights](https://github.com/ForecastOS/investos/tree/v0.6.3/investos/portfolio/strategy/spo_weights.py) (Single Period Optimization Weights)
    -   Like SPO, but performs optimization using target weights instead of expected returns
    -   Uses [CVXPY](https://www.cvxpy.org/tutorial/intro/index.html)
    - **Optimization**: minimizes deviation from target weights, while adhering to constraints

This guide will explain how to use these classes.

## RankLongShort

To use the `RankLongShort` strategy, you will need:

-   actual_returns: pd.DataFrame
-   `metric_to_rank`: pd.DataFrame

Optional instantiation options you may need:

-   `n_periods_held`: int = 1
-   leverage: float = 1
-   ratio_long: float = 1
-   ratio_short: float = 1
-   percent_short: float = 0.25
-   percent_long: float = 0.25
-   costs: [BaseCost] = []

Here's a simple instantiation example:

```python
from investos.portfolio.strategy import RankLongShort

actual_returns_df = pd.DataFrame(...) # Assets as columns, datetimes as index
metric_to_rank_df = pd.DataFrame(...) # Assets as columns, datetimes as index

strategy = RankLongShort(
    actual_returns=actual_returns_df,
    metric_to_rank=metric_to_rank_df,
    leverage=1.5,
    ratio_long=100,
    ratio_short=50,
)
```

Once instantiated, you can use the `generate_trade_list` method to get trades for a given datetime:

```python
holdings = pd.Series(...) # Indexed by asset
t = dt.datetime.today()

trades = strategy.generate_trade_list(holdings, t)
```

You can also plug the strategy into BacktestController to run a full backtest!

```python
import investos as inv

backtest_controller = inv.portfolio.BacktestController(
    strategy=strategy
)
```

## SPO

To use the `SPO` strategy, you will need:

-   actual_returns: pd.DataFrame
-   forecast_returns: pd.DataFrame

Optional instantiation options you may need:

-   costs: [BaseCost] = []
-   constraints: [BaseConstraint] = [LongOnlyConstraint(), MaxWeightConstraint()]
-   risk_model: BaseRisk = None
-   solver=cvx.OSQP
-   solver_opts=None
-   `cash_column_name`="cash"

Here's a simple instantiation example:

```python
actual_returns_df = pd.DataFrame(...) # Assets as columns, datetimes as index
forecast_returns_df = pd.DataFrame(...) # Assets as columns, datetimes as index

strategy = SPO(
    actual_returns=actual_returns_df,
    forecast_returns=forecast_returns_df
)
```

Like RankLongShort, or any other InvestOS investment strategy, once instantiated, you can use the `generate_trade_list` method to get trades for a given datetime:

```python
holdings = pd.Series(...)
t = dt.datetime.today()

trades = strategy.generate_trade_list(holdings, t)
```

and plug the strategy into BacktestController to run a full backtest!

```python
import investos as inv

backtest_controller = inv.portfolio.BacktestController(
    strategy=strategy
)
```

For SPO specifically, if the optimization problem is unbounded, infeasible, or if there's an error with the solver, the `generate_trade_list` method will return a zero trade for all holdings for the given `t` datetime.

## SPOTranches

The `SPOTranches` strategy uses all of the same arguments as `SPO`. It also uses one additional (optional) argument for determining the holding period for each optimized tranche:

-   `n_periods_held`: integer = 5

## SPOWeights

The `SPOWeights` strategy uses most of the same arguments as `SPO`. 

It doesn't using the following argument from `SPO`:

-   forecast_returns: pd.DataFrame

It uses the following additional argument instead:

-   target_weights: pd.DataFrame

## Next: The Choice Is Yours

Want to explore creating your own custom investment strategy? Check out [Custom Investment Strategies](/guides/bespoke/custom_investment_strategies).

Want to learn more about using cost models? Check out [Cost Models](/guides/off_the_shelf/cost_models).
