<h1>How InvestOS Works</h1>

## The Pieces

InvestOS has the following classes, which work together (as will be described shortly) to create portfolios and associated backtest results:

-   [BacktestController](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/backtest_controller.py)
-   [BaseStrategy](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/strategy/base_strategy.py)
-   [BaseResult](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/result/base_result.py)
-   [BaseCost](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/cost_model/base_cost.py)
-   [BaseConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/base_constraint.py)
-   [BaseRisk](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/risk_model/base_risk.py)

### BacktestController

[BacktestController](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/backtest_controller.py) incrementally generates point-in-time portfolio positions based on an investment BaseStrategy.

Incrementally generated positions are saved into a BaseResult class, which contains a myriad of performance reporting methods for convenience.

### BaseStrategy

[BaseStrategy](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/strategy/base_strategy.py) provides a common interface to extend to create custom investment strategies.

BacktestController will ask (BaseStrategy) investment strategies to `generate_trade_list` for each point-in-time period in your backtest. BacktestController will then save the trade list returned by the (BaseStrategy) investment strategy, calculate resulting portfolio holdings, and save both into BaseResult for performance reporting.

Off-the-shelf investment strategies, which extend BaseStrategy, include:

-   [Single Period Optimization](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/strategy/spo.py) (SPO): optimizes for max estimated return after estimated costs and (a utility penalty for) estimated portfolio variance
-   [Single Period Optimization Tranches](https://github.com/ForecastOS/investos/tree/v0.4.1/investos/portfolio/strategy/spo_tranches.py) (SPO Tranches): Like SPO, but builds portfolio in separate constrained / optimized tranches. Tranches are cycled in and out by customizable holding period. Tranches can be analyzed and altered in flight using BacktestController hooks.
-   [RankLongShort](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/strategy/rank_long_short.py)

**Note**: `generate_trade_list` can be used to generate a trade list outside of a backtest context (i.e. to implement your investment strategy in the market).

**Note**: we are starting work on new SPO classes for maximizing expected Sharpe ratio (instead of return).

### BaseResult

The [BaseResult](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/result/base_result.py) class captures trades and resulting portfolio positions sent from BacktestController.

It provides performance reporting methods for convenience, allowing you to analyze your backtest results.

### BaseCost

[BaseCost](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/cost_model/base_cost.py) provides a common interface to extend to create custom cost models.

Cost models are passed into your investment strategy (BaseStrategy) upon initialization of your investment strategy. Your investment strategy will calculate estimated (if using SPO) and simulated realized costs, based on the logic in your cost model.

Off-the-shelf costs, which extend BaseCost, include:

-   [ShortHoldingCost](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/cost_model/short_holding_cost.py)
-   [TradingCost](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/cost_model/trading_cost.py)

### BaseConstraint

[BaseConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/base_constraint.py) provides a common interface to extend to create custom constraint models.

**Constraint models are only useful if your investment strategy uses convex portfolio optimization** (used by SPO classes).

Constraint models are passed into your investment strategy (BaseStrategy) upon initialization of your investment strategy. Your investment strategy will optimize your trades and resulting positions without breaching any of the constraints you define (e.g. max leverage, max weight, equal long / short, etc.).

If your constraints are overly restrictive (preventing a possible solution), the BacktestController will default to a zero trade (i.e. will hold starting positions with no changes).

Off-the-shelf constraints, which extend BaseConstraint, include:

-   [Factor constraints](https://github.com/ForecastOS/investos/tree/v0.4.1/investos/portfolio/constraint_model/factor_constraint.py)
-   [Leverage constraints](https://github.com/ForecastOS/investos/tree/v0.4.1/investos/portfolio/constraint_model/leverage_constraint.py)
-   [Long / market-neutral constraints](https://github.com/ForecastOS/investos/tree/v0.4.1/investos/portfolio/constraint_model/long_constraint.py)
-   [Position weight (max/min) constraints](https://github.com/ForecastOS/investos/tree/v0.4.1/investos/portfolio/constraint_model/weight_constraint.py)
-   [Turnover constraints](https://github.com/ForecastOS/investos/tree/v0.4.1/investos/portfolio/constraint_model/trade_constraint.py)

**There are +20 off-the-shelf contraint models available.** We regularly release new constraint models as needed / helpful.

### BaseRisk

[BaseRisk](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/risk_model/base_risk.py) extends BaseCost.

Unlike BaseCost, it does not apply actual costs to your backtest results (BaseResult); realized costs from risk models in your backtest will always be 0.

It does, however:

1. Optionally apply a (utility) cost during portfolio creation for convex-optimization-based investment strategies (like [SPO](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/strategy/spo.py)) to penalize estimated portfolio volatility
2. Optionally output a portfolio variance estimate for creating a mean-variance optimized (MVO) portfolio (i.e. minimizing variance for a given return)

## Extend Base Classes For Custom Use Cases

With the exception of BacktestController, we expect you to extend the above base classes to fit your own use cases (where needed). Following guides will expand on how to customize each class above.

If this is of interest, we also encourage you to review the open-source codebase; we've done our best to make it as simple and understandable as possible. Should you extend one of our base classes in a way that might be useful to other investors, we also encourage you to open a PR!

## Use Off-The-Shelf Classes Where Possible

As mentioned above, we've created some off-the-shelf classes that extend the above base classes - like [SPO](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/strategy/spo.py) (single period optimization), an extension of BaseStrategy.

Hopefully you find them useful and they save you time. Our goal is to cover 100% of common backtesting and portfolio engineering requirements with our off-the-shelf models.

Common off-the-shelf classes will be discussed in more detail in the following guides!

## Next: An Example Backtest Using RankLongShort

Now that you have an idea how InvestOS works, let's move on to our next guide: [Rank Long Short](/guides/simple_examples/rank_long_short).
