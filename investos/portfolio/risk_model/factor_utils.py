

def get_t_statistics(model, X, y):
    """Calculate factor t-statistics value

    Parameters
    ----------
    model : sklearn.linear_model._base.LinearRegression
        Linear regression model fits a linear model
    X : pd.DataFrame
        N*K training data with all factors value.
    y : pd.Series
        N*1 target values

    Returns
    -------
    t_values
    array
        The t-statistics for all the factors
    """
    predictions = model.predict(X)
    residuals = y - predictions
    # Calculate the residual standard error
    rss = (residuals ** 2).sum()
    n = len(y)  # number of sample
    p = X.shape[1]  # number of factors
    rmse = (rss / n) ** 0.5
    rse = (rss / (n - p - 1)) ** 0.5
    # Calculate t-values for coefficients
    t_values = model.coef_ / (rse / n ** 0.5)

    return t_values