<h1>Using Off-The-Shelf Constraint Models</h1>

## Usage

Like cost models, using constraint models is also easy!

You simply pass instantiated models into your desired convex optimization investment strategy:

```
from investos.portfolio.strategy import SPO

strategy = SPO(
    actual_returns = actual_returns,
    ...,
    constraints=[
        ConstraintModelA(*args, **kwargs),
        ConstraintModelB(*args, **kwargs)
    ],
)
```

and that's it!

## Optional Arguments

All constraint models take the following optional arguments:

-   exclude_assets: [str]
-   include_assets: [str]
    -   Can't be used alongside exclude_assets

---

InvestOS provides the following constraint models:

## Factor Constraints, Portfolio

### ZeroFactorExposureConstraint

[ZeroFactorExposureConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/factor_constraint.py) enforces 0 net portfolio weight in a given factor.

Instantiate with the following arguments:

-   `factor_exposure`: pd.DataFrame (asset ID columns and datetime index) or pd.Series (asset ID index, if same value across datetime)

### MaxFactorExposureConstraint

[MaxFactorExposureConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/factor_constraint.py) enforces max portfolio weight in a given factor.

Instantiate with the following arguments:

-   limit: float = 0.05
-   `factor_exposure`: pd.DataFrame (asset ID columns and datetime index) or pd.Series (asset ID index, if same value across datetime)

### MinFactorExposureConstraint

[MinFactorExposureConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/factor_constraint.py) enforces min portfolio weight in a given factor.

Instantiate with the following arguments:

-   limit: float = -0.05
-   `factor_exposure`: pd.DataFrame (asset ID columns and datetime index) or pd.Series (asset ID index, if same value across datetime)

### MaxAbsoluteFactorExposureConstraint

[MaxAbsoluteFactorExposureConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/factor_constraint.py) enforces max absolute portfolio weight in a given factor.

Instantiate with the following arguments:

-   limit: float = 0.4
-   `factor_exposure`: pd.DataFrame (asset ID columns and datetime index) or pd.Series (asset ID index, if same value across datetime)

## Factor Constraints, Trade

### ZeroTradeFactorExposureConstraint

[ZeroTradeFactorExposureConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/factor_constraint.py) enforces 0 net **trade** weight in a given factor.

Instantiate with the following arguments:

-   `factor_exposure`: pd.DataFrame (asset ID columns and datetime index) or pd.Series (asset ID index, if same value across datetime)

### MaxTradeFactorExposureConstraint

[MaxTradeFactorExposureConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/factor_constraint.py) enforces max **trade** weight in a given factor.

Instantiate with the following arguments:

-   limit: float = 0.05
-   `factor_exposure`: pd.DataFrame (asset ID columns and datetime index) or pd.Series (asset ID index, if same value across datetime)

### MinTradeFactorExposureConstraint

[MinTradeFactorExposureConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/factor_constraint.py) enforces min **trade** weight in a given factor.

Instantiate with the following arguments:

-   limit: float = -0.05
-   `factor_exposure`: pd.DataFrame (asset ID columns and datetime index) or pd.Series (asset ID index, if same value across datetime)

### MaxAbsoluteTradeFactorExposureConstraint

[MaxAbsoluteTradeFactorExposureConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/factor_constraint.py) enforces max absolute **trade** weight in a given factor.

Instantiate with the following arguments:

-   limit: float = 0.01
-   `factor_exposure`: pd.DataFrame (asset ID columns and datetime index) or pd.Series (asset ID index, if same value across datetime)

## Leverage Constraints, Portfolio

### MaxLeverageConstraint

[MaxLeverageConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/leverage_constraint.py) enforces a limit on the (absolute) leverage of the portfolio.

E.g. For leverage of 2.0x, a portfolio with 100MM net value (i.e. the portfolio value if it were converted into cash, ignoring liquidation / trading costs) could have 200MM of (combined long and short) exposure.

To instantiate MaxLeverageConstraint you will need to set the following argument:

-   limit: float = 1.0

### MaxShortLeverageConstraint

[MaxShortLeverageConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/leverage_constraint.py#L46C6-L46C6) enforces a limit on the short leverage of the portfolio.

To instantiate MaxShortLeverageConstraint you will need to set the following argument:

-   limit: float = 1.0

### MaxLongLeverageConstraint

[MaxLongLeverageConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/leverage_constraint.py#L81) enforces a limit on the long leverage of the portfolio.

To instantiate MaxLongLeverageConstraint you will need to set the following argument:

-   limit: float = 1.0

## Leverage Constraints, Trade

### MaxLongTradeLeverageConstraint

[MaxLongTradeLeverageConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/leverage_constraint.py) enforces a max on the long leverage of the trade.

Instantiate with the following arguments:

-   limit: float = 0.025

### MaxShortTradeLeverageConstraint

[MaxShortTradeLeverageConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/leverage_constraint.py) enforces a max on the short leverage of the trade.

Instantiate with the following arguments:

-   limit: float = 0.025

## Long Constraints

### LongOnlyConstraint

[LongOnlyConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/long_constraint.py#L4) enforces no short positions.

### LongCashConstraint

[LongCashConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/long_constraint.py#L43) enforces no short cash positions.

To instantiate LongCashConstraint you will need to set the following argument:

-   include_assets: [str] = ["cash"]

### EqualLongShortConstraint

[EqualLongShortConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/long_constraint.py#L82) enforces equal long and short exposure.

To instantiate EqualLongShortConstraint you will need to set the following argument:

-   exclude_assets: [str] = ["cash"]

## Weight Constraints

### MaxWeightConstraint

[MaxWeightConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/weight_constraint.py#L4) enforces a limit (as a percentage) on the weight of each asset in your portfolio.

To instantiate MaxWeightConstraint you will need to set the following arguments:

-   limit: float = 0.025
-   exclude_assets: [str] = ["cash"]

### MinWeightConstraint

[MinWeightConstraint](https://github.com/ForecastOS/investos/tree/v0.3.9/investos/portfolio/constraint_model/weight_constraint.py#L53) enforces a limit (as a percentage) on the weight of each asset in your portfolio.

To instantiate MinWeightConstraint you will need to set the following arguments:

-   limit: float = -0.025
-   exclude_assets: [str] = ["cash"]

## A Quick Note

If you aren't using a convex optimization based investment strategy (like SPO), constraint models don't do anything.

## Next: Custom Constraint Models

Next, let's explore building our own constraint models: [Custom Constraint Models](/guides/bespoke/custom_constraint_models).
