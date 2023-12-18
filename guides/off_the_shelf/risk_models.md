<h1>Using Off-The-Shelf Risk Models</h1>

## Usage

Using risk models is simple!

You simply pass instantiated risk model instances into your desired investment strategy:

```
import investos as inv
from investos.portfolio.risk_model import *

risk_model = FactorRisk(
    factor_covariance=df_factor_covar,
    factor_loadings=df_loadings,
    idiosyncratic_variance=df_idio,
    exclude_assets=["cash"]
)

strategy = inv.portfolio.strategy.SPO(
    actual_returns=df_actual_returns,
    ...
    risk_model=risk_model,
)
```

and that's it!

_Note: some simple investment strategies do not support risk models. If you pass a risk model to one of these strategies, it will have no effect._

## Optional Arguments

All risk models take the following optional arguments:

-   `exclude_assets`: [str]
-   `include_assets`: [str]
    -   Can't be used with exclude_assets
-   `gamma`: float = 1
    -   Linearly increases utility penalty during convex optimization trade list generation
    -   Increase to penalize risk more, decrease to penalize risk less

---

InvestOS provides the following risk models:

## FactorRisk

[FactorRisk](https://github.com/ForecastOS/investos/tree/v0.3.10/investos/portfolio/risk_model/factor_risk.py) is a multi-factor risk model.

To instantiate FactorRisk you will need to set the following arguments:

-   `factor_covariance`: pd.DataFrame
    -   Columns and index keys should be risk factors
    -   Values are covariances
    -   Optionally: with date as the first key in multi-index dataframe; to allow risk estimates to change throughout time. By default, will look for risk date equal to or less than trade date
-   `factor_loadings`: pd.DataFrame
    -   Columns should be unique asset IDs
    -   Index keys should be risk factors
    -   Values are loadings
    -   Optionally: with date as the first key in multi-index dataframe; to allow risk estimates to change throughout time. By default, will look for risk date equal to or less than trade date
-   `idiosyncratic_variance`: pd.DataFrame | pd.Series
    -   Columns should be unique asset IDs
    -   Values are idiosyncratic risks (residuals to factor risk)
    -   Optionally: with date as index in dataframe; to allow risk estimates to change throughout time. By default, will look for risk date equal to or less than trade date

## StatFactorRisk

[StatFactorRisk](https://github.com/ForecastOS/investos/tree/v0.3.10/investos/portfolio/risk_model/stat_factor_risk.py) creates a PCA-factor based risk model from `actual_returns`. To use this model, there must be more periods in `actual_returns` than assets in your investment strategy.

To instantiate StatFactorRisk you will need to set the following arguments:

-   `actual_returns`: pd.DataFrame

You may optionally set the following arguments:

-   `n_factors`: integer = 5
-   `start_date`: datetime = actual_returns.index[0]
-   `end_date`: datetime = actual_returns.index[-1]
-   `recalc_each_i_periods`: integer|boolean = False
-   `timedelta`: pd.Timedelta = pd.Timedelta("730 days")
    -   Lookback period for calculating risk from actual_returns

## Next: The Choice Is Yours

Want to explore an end-to-end example? Check out [Single Period Optimization](/guides/simple_examples/spo).

Want to learn more about what guides are coming next? Check out [Guides Under Development](/guides/coming_soon/guides_under_development).
