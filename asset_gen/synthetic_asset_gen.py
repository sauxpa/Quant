#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import random as rd
from numpy import linalg
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from typing import Tuple

class Return_Normal_Gen():
    """Generates sample returns and prices from returns drift, volatility and correation.
    """
    def __init__(self, 
                 corr_matrix: np.ndarray,
                 vol: np.ndarray,
                 drift: np.ndarray,
                 n_steps: int=1.0, 
                ) -> None:
        
        self._n_assets = corr_matrix.shape[0]
        
        # check corr_matrix is a 2D square matrix
        assert len(corr_matrix.shape) == 2 and corr_matrix.shape[0] == corr_matrix.shape[1]
        # check vol and drift have same dim as corr_matrix
        assert len(vol) == corr_matrix.shape[0]
        assert len(drift) == corr_matrix.shape[0]

        self._drift = drift
        self._vol = vol
        self._corr_matrix = corr_matrix
        self._cov_matrix = np.matmul(np.diag(vol), np.matmul(corr_matrix, np.diag(vol)))        
        
        self._n_steps = n_steps

    @property
    def corr_matrix(self) -> np.ndarray:
        return self._corr_matrix
    @corr_matrix.setter
    def corr_matrix(self, new_corr_matrix) -> None:
        self._corr_matrix = new_corr_matrix
    
    @property
    def vol(self) -> np.ndarray:
        return self._vol
    @vol.setter
    def vol(self, new_vol) -> None:
        self._vol = new_vol
        
    @property
    def drift(self) -> np.ndarray:
        return self._drift
    @drift.setter
    def drift(self, new_drift) -> None:
        self._drift = new_drift
        
    @property
    def n_steps(self) -> float:
        return self._n_steps
    @n_steps.setter
    def n_steps(self, new_n_steps) -> None:
        self._n_steps = new_n_steps
    
    @property
    def n_assets(self) -> int:
        return self.corr_matrix.shape[0]
    
    @property
    def cov_matrix(self) -> np.ndarray:
        return np.matmul(np.diag(self.vol), np.matmul(self.corr_matrix, np.diag(self.vol)))        
        
    @property
    def rets(self) -> np.ndarray:
        return self._rets
    
    def gen_rets(self) -> np.ndarray:
        self._rets = rd.multivariate_normal(self.drift, self.cov_matrix, self.n_steps)
        return self._rets
    
    @property
    def corr_eigen(self) -> Tuple[np.ndarray, np.ndarray]:
        corr_eigenvalues, corr_eigenvectors = linalg.eig(corr_matrix)
        return np.real(corr_eigenvalues), corr_eigenvectors

    @property
    def prices(self) -> np.ndarray:
        return np.cumprod(self.rets+1, 0)
        
    def plot(self) -> None:
        fig, axes = plt.subplots(nrows=1, ncols=2)

        ax = axes[0]
        ax.plot(self.prices)

        ax = axes[1]
        ax.plot(self.rets)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.2%}'.format(x)))

        plt.tight_layout()
        plt.show()