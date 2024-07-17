import numpy as np
import pandas as pd


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
            self.first_level_index = df_idio_returns.index[self.window-1:]
        else:
            self.window = 0
        self.recalc_freq = recalc_freq        
        self._df_idio_returns = df_idio_returns.set_index(['datetime','id']).sort_values(by='datetime')
        self.first_level_index = self._df_idio_returns.index.get_level_values(0).unique()[[idx for idx in range(len(self._df_idio_returns.index.get_level_values(0).unique()) - 1, self.window, -self.recalc_freq)]]

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
            cov_matrices[idx]['datetime'] = self.first_level_index[idx]
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

    def calculate_ewma_idiosyncratic_variance(self,half_life: int= 360):

        cov_matrices = []
        if not self.window:

            cov_matrices.append(self.idio_var_emwa(df_idio_returns = self._df_idio_returns,half_life = half_life))            
        else:
            for date in self.first_level_index:
                cov_matrices.append(self.idio_var_emwa(df_idio_returns = self._df_idio_returns.loc[date - pd.Timedelta(days = self.window):date,:],half_life = half_life))

        self._cov_matrices = cov_matrices
        return self.convert_cov_matrices(cov_matrices)
