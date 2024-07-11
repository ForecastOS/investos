import numpy as np
import pandas as pd


class AssetDigonalVarAdjuster:

    def __init__(self, df_idio_returns, window=None):
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
            #self.first_level_index = df_idio_returns.index
            self.window = 0
        
        self._df_idio_returns = df_idio_returns
        self.second_level_index = df_idio_returns.id.unique()
        self.first_level_index = df_idio_returns.index[self.window-1:]
    @property
    def ids(self):
        return self._second_level_index

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
        #print(cov_matrices)
        #cov_matrices = np.array(cov_matrices)
        # for idx in self.first_level_index:
        #     cov_matrices[idx]['datetime'] = self.first_level_index[idx]
        #     cov_matrices[idx].set_index('datetime', append = True, inplace = True)
        #print(len(self.second_level_index))
        #multi_index = pd.MultiIndex.from_product([self.first_level_index, self.second_level_index], names=['datetime', 'factor'])
        #print(cov_matrices[0].shape)
        #return pd.DataFrame(cov_matrices.reshape((-1, cov_matrices.shape[-1])), index = multi_index)
        for idx in range(len(cov_matrices)):
            cov_matrices[idx]['datetime'] = self.first_level_index[idx]
            cov_matrices[idx] = cov_matrices[idx].reset_index().set_index(['datetime','id'])
        return pd.concat(cov_matrices)


    def idio_var_emwa(self, df_idio_returns, half_life: int = None):
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

    def calc_idio_var_raw(self,half_life):

        cov_matrices = []
        if not self.window:

            cov_matrices.append(self.idio_var_emwa(self._df_idio_returns,half_life))            
        else:
            for date in self.first_level_index:

                cov_matrices.append(self.idio_var_emwa(self._df_idio_returns.loc[date - pd.Timedelta(days = self.window):date,:],half_life))

        self._cov_matrices = cov_matrices
        return self.convert_cov_matrices(cov_matrices)
