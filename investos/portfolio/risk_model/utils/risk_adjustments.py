import numpy as np
import pandas as pd
from investos.portfolio.risk_model.utils.risk_utils import *

class FactorCovAdjuster:
    """Adjustments on factor covariance matrix"""

    def __init__(self, factor_return_matrix: pd.DataFrame, recalc_freq: int, window: int | None = None) -> None:
        """Initialization

        Parameters
        ----------
        factor_return_matrix : np.ndarray
            Factor return matrix (num_of_dates * num_of_factors)
        recalc_freq: int

        window: int

        """
        self.length_of_dates, self.num_of_factors = factor_return_matrix.shape
        if self.num_of_factors > self.length_of_dates:
            raise Exception("number of periods must be larger than number of factors")
        self.factor_return_matrix = factor_return_matrix.astype("float64")
        if window and window > self.length_of_dates:
            raise Exception("number of dates must be larger than window size")

        if window:
            self.window = window
        else:
            self.window = 0
        self.factor_cov_matrix = None
        self.recalc_freq = recalc_freq
        self.factor_index = factor_return_matrix.columns
        self.datetime_index = factor_return_matrix.index[[idx for idx in range(len(factor_return_matrix)-1,self.window,-self.recalc_freq)]]

    def convert_cov_matrices(self,cov_matrices: list) -> pd.DataFrame:
        """Convert the list of covariance matrix (3D) to 2D pandas DataFrame

        Parameters
        ----------
        cov_matrices : list
            A list of covariance matrix (one or more depends on the window)

        Returns
        -------
        pd.DataFrame
            cov_matrices, 2D pandas.DataFrame with 2 level of index (datetime, factor)
        """
        cov_matrices = np.array(cov_matrices)
        multi_index = pd.MultiIndex.from_product([self.datetime_index, self.factor_index], names=['datetime', 'factor'])
        return pd.DataFrame(cov_matrices.reshape((-1, cov_matrices.shape[-1])), index = multi_index, columns = self.factor_index)


    def est_factor_cov_matrix_raw(self, factor_return_matrix: pd.DataFrame ,half_life: int | None = None) -> pd.DataFrame:
        """Calculate the factor covariance matrix, factor_cov_matrix (K*K)

        Parameters
        ----------
        factor_return_matrix: pd.DataFrame
            Factor return matrix (T*K)
        half_life : int
            Steps it takes for weight in EWA to reduce to half of the original value

        Returns
        -------
        np.ndarray
            factor_cov_matrix, denoted by `F_Raw`
        """
        cov_matrices = []
        factor_return_matrix = factor_return_matrix.astype("float64")
        if not self.window:
            cov_matrices.append(calc_ewma_cov(factor_return_matrix.T.to_numpy(), half_life))
        else:
            for idx in range(len(factor_return_matrix)-1,self.window, -self.recalc_freq):
                cov_matrices.append(calc_ewma_cov(factor_return_matrix.iloc[idx-self.window:idx,:].T.to_numpy(), half_life))

        self.factor_cov_matrix = self.convert_cov_matrices(cov_matrices)
        return self.factor_cov_matrix


    def newey_west_adjust(self,factor_cov_matrix,
        factor_return_matrix: np.ndarray, half_life: int, max_lags: int, multiplier: float) -> np.ndarray:
        """Apply Newey-West adjustment on `F_Raw`

        Parameters
        ----------
        half_life : int
            Steps it takes for weight in EWA to reduce to half of the original value
        max_lags : int
            Maximum Newey-West correlation lags
        multiplier : int
            Number of periods a factor_cov_matrix with new frequence contains

        Returns
        -------
        np.ndarray
            Newey-West adjusted factor_cov_matrix, denoted by `F_NW`
        """
        for d in range(1, max_lags + 1):
            cov_pos_delta = calc_ewma_cov(factor_return_matrix, half_life, d)
            factor_cov_matrix += (1 - d / (1 + max_lags)) * (cov_pos_delta + cov_pos_delta.T)
        eigen_vals, eigen_vecs = np.linalg.eigh(factor_cov_matrix * multiplier)
        eigen_vals[eigen_vals <= 0] = 1e-14        # adjust diagonal matrix to be positive definite
        factor_cov_matrix = eigen_vecs.dot(np.diag(eigen_vals)).dot(eigen_vecs.T)
        eigen_vals, eigen_vecs = np.linalg.eigh(factor_cov_matrix)
        return factor_cov_matrix

    def apply_newey_west_adjustment(self, max_lags:int = 1, multiplier: float = 1.2, half_life: int | None = 480,
                            ) -> pd.DataFrame:

        cov_matrices = []
        if not self.window:
            cov_matrices.append(self.newey_west_adjust(
                                                  factor_cov_matrix = self.factor_cov_matrix,
                                                  factor_return_matrix = self.factor_return_matrix.T.to_numpy(),
                                                  half_life = half_life,
                                                  max_lags = max_lags,
                                                  multiplier = multiplier))
        else:
            for idx,val in enumerate(self.factor_cov_matrix.index.get_level_values(0).unique()):

                cov_matrices.append(self.newey_west_adjust( factor_cov_matrix = self.factor_cov_matrix.loc[val].T.to_numpy(),
                                                            factor_return_matrix = self.factor_return_matrix.iloc[self.factor_return_matrix.shape[0]-idx*self.recalc_freq- self.window:self.factor_return_matrix.shape[0] - idx*self.recalc_freq,:].T.to_numpy(),
                                                            half_life = half_life,
                                                            max_lags = max_lags,
                                                            multiplier = multiplier))
        self.factor_cov_matrix = self.convert_cov_matrices(cov_matrices)
        return self.factor_cov_matrix

    def eigenfactor_risk_adjust(self, factor_cov_matrix, coef: float, window: int, num_sim: int = 1000) -> np.ndarray:
        """Apply eigenfactor risk adjustment on `F_NW`

        Parameters
        ----------
        coef : float
            Adjustment coefficient
        M : int, optional
            Times of Monte Carlo simulation, by default 1000

        Returns
        -------
        np.ndarray
            Eigenfactor risk adjusted factor_cov_matrix, denoted by `F_Eigen`
        """

        eigen_vals_0, eigen_vecs_0 = np.linalg.eigh(factor_cov_matrix)          # F_NW
        eigen_vals_0[eigen_vals_0 <= 0] = 1e-14                                 # adjust diagonal matrix to be positive definite
        sim_risk_deviation = np.zeros((self.num_of_factors,))
        for _ in range(num_sim):                                                # number of MC simulation
            sim_eigen_factor_matrix = np.array([np.random.normal(0, var**0.5, window) for var in eigen_vals_0])
            sim_factor_return_matrix = eigen_vecs_0.dot(sim_eigen_factor_matrix)
            sim_factor_return_cov = sim_factor_return_matrix.dot(sim_factor_return_matrix.T) / (window- 1)
            sim_eigen_vals, sim_eigen_vecs = np.linalg.eigh(sim_factor_return_cov)
            sim_eigen_vals[sim_eigen_vals <= 0] = 1e-14                         # adjust diagonal matrix to be positive definite
            sim_real_eigen_vals = sim_eigen_vecs.T.dot(factor_cov_matrix).dot(sim_eigen_vecs)
            sim_risk_deviation += np.diag(sim_real_eigen_vals) / sim_eigen_vals
        sim_risk_deviation[sim_risk_deviation <= 0] = 1e-14                     # adjust diagonal matrix to be positive definite
        sim_risk_deviation = np.sqrt(sim_risk_deviation / num_sim)
        adj_sim_risk_deviation = coef * (sim_risk_deviation - 1.0) + 1.0
        eigen_vals_0_tilde = adj_sim_risk_deviation**2 * eigen_vals_0
        factor_cov_matrix = eigen_vecs_0.dot(np.diag(eigen_vals_0_tilde)).dot(eigen_vecs_0.T)
        return factor_cov_matrix

    def apply_eigenfactor_risk_adjustment(self, max_lags:int = 1, multiplier: float = 1.2, half_life: int | None = 480,
                                  coef: float = 1.2, num_of_sim: int = 1000, window: int = 480) -> pd.DataFrame:

        cov_matrices = []
        if not self.window:
            cov_matrices.append(self.eigenfactor_risk_adjust(self.factor_cov_matrix, coef, window,num_of_sim))
        else:
            for _,val in enumerate(self.factor_cov_matrix.index.get_level_values(0).unique()):
                cov_matrices.append(self.eigenfactor_risk_adjust(self.factor_cov_matrix.loc[val].T.to_numpy(), coef, window,num_of_sim))

        self.factor_cov_matrix = self.convert_cov_matrices(cov_matrices)
        return self.factor_cov_matrix

    def volatility_regime_adjust(self,factor_cov_matrix, factor_return_matrix: np.ndarray, half_life: int) -> np.ndarray:
        """Apply volatility regime adjustment on `F_Eigen`

        Parameters
        ----------
        factor_cov_matrix : np.ndarray
            Previously estimated factor covariance matrix (last `F_Eigen`, since `F_VRA`
            could lead to huge fluctuations) on only one period (not aggregated); the
            order of factors should remain the same
        half_life : int
            Steps it takes for weight in EWA to reduce to half of the original value

        factor_return_matrix: np.ndarray
            Factor return matrix (T*K)
        Returns
        -------
        np.ndarray
            Volatility regime adjusted factor_cov_matrix, denoted by `F_VRA`
        """
        factor_std = np.sqrt(np.diag(factor_cov_matrix))
        bias_stats = BiasStatsCalculator(factor_return_matrix, factor_std).single_window(half_life)   # need to be fixed: using rolling window
        factor_cov_matrix = factor_cov_matrix * (bias_stats**2)#.mean(axis=0)  # Lambda^2
        return factor_cov_matrix

    def apply_volatility_regime_adjustment(self,  half_life: int | None = 480 ):

        cov_matrices = []
        if not self.window:
            cov_matrices.append(self.volatility_regime_adjust(self.factor_cov_matrix, self.factor_return_matrix.T.to_numpy(),half_life))
        else:
            for idx,val in enumerate(self.factor_cov_matrix.index.get_level_values(0).unique()):
                cov_matrices.append(self.volatility_regime_adjust(self.factor_cov_matrix.loc[val].T.to_numpy(), self.factor_return_matrix.iloc[self.factor_return_matrix.shape[0]-idx*self.recalc_freq- self.window:self.factor_return_matrix.shape[0] - idx*self.recalc_freq,:].T.to_numpy(),half_life))

        self.factor_cov_matrix = self.convert_cov_matrices(cov_matrices)
        return self.factor_cov_matrix

class AssetDigonalVarAdjuster:

    def __init__(self, df_idio_returns,window: int | None = 251, recalc_freq: int = 21):
        """Initialization

        Parameters
        ----------
        df_loadings: pd.DataFrame
        df_factor_returns: pd.DataFrame

        Return
        ----------
        IVM (Idiosyncratic Variance Matrix (IVM): pd.DataFrame
         A matrix that contains the variances of the idiosyncratic (or specific) risks of different assets. This matrix is typically diagonal, where each diagonal element represents the variance of the idiosyncratic risk of an asset.
        """

        if window:
            self.window = window
            self.datetime_index = df_idio_returns.index[self.window-1:]
        else:
            self.window = 0
        self.recalc_freq = recalc_freq
        self._df_idio_returns = df_idio_returns.set_index(['datetime','id']).sort_values(by='datetime')
        self.datetime_index = self._df_idio_returns.index.get_level_values(0).unique()[[idx for idx in range(len(self._df_idio_returns.index.get_level_values(0).unique()) - 1, self.window, -self.recalc_freq)]]

    def convert_cov_matrices(self,cov_matrices: list) -> pd.DataFrame:
        """Convert the list of covariance matrix (3D) to 2D pandas DataFrame

        Parameters
        ----------
        cov_matrices : list
            A list of covariance matrix (one or more depends on the window)

        Returns
        -------
        pd.DataFrame
            cov_matrices, 2D pandas.DataFrame with 2 level of index (datetime, factor)
        """
        for idx in range(len(cov_matrices)):
            cov_matrices[idx]['datetime'] = self.datetime_index[idx]
            cov_matrices[idx] = cov_matrices[idx].reset_index().set_index(['datetime','id'])
        return pd.concat(cov_matrices)


    def idio_var_emwa(self, df_idio_returns, half_life: int = 360):
        """
        Calculate the EWMA idiosyncratic return diagonal variance matrix.

        Parameters:
        ----------
        df_idio_returns : pd.DataFrame
        DataFrame containing columns 'datetime', 'id', and 'returns'.

        half_life : int
        Half-life for the EWMA calculation.

        Returns:
        -------
        pd.DataFrame: Diagonal variance matrix.
        """
        # Convert datetime to pandas datetime format if not already
        df_idio_returns_cp = df_idio_returns.reset_index().copy()
        df_idio_returns_cp.loc[:,'datetime'] = pd.to_datetime(df_idio_returns_cp['datetime'])

        # Sort by datetime for proper calculation
        df_idio_returns_cp = df_idio_returns_cp.sort_values(by='datetime')

        # Calculate the decay factor for EWMA
        decay_factor = np.log(2) / half_life

        # Calculate weights for EWMA
        df_idio_returns_cp.loc[:,'weight'] = np.exp(-decay_factor * (df_idio_returns_cp['datetime'].max() - df_idio_returns_cp['datetime']).dt.days)

        # Calculate weighted returns squared
        df_idio_returns_cp.loc[:,'weighted_squared_returns'] = df_idio_returns_cp['weight'] * df_idio_returns_cp['factor_return_1d_error']**2

        # Calculate the EWMA variance for each 'id'
        ewma_variance = df_idio_returns_cp.groupby('id').apply(lambda x: x['weighted_squared_returns'].sum() / x['weight'].sum())

        return pd.DataFrame(ewma_variance, index = ewma_variance.index, columns = ['idio_variance'])    # this is variance

    def est_ewma_idiosyncratic_variance(self,half_life: int= 360):

        cov_matrices = []
        if not self.window:

            cov_matrices.append(self.idio_var_emwa(df_idio_returns = self._df_idio_returns,half_life = half_life))
        else:
            for date in self.datetime_index:
                cov_matrices.append(self.idio_var_emwa(df_idio_returns = self._df_idio_returns.loc[date - pd.Timedelta(days = self.window):date,:],half_life = half_life))

        self._cov_matrices = cov_matrices
        return self.convert_cov_matrices(cov_matrices)
