#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import random as rd

def random_corr_matrix_from_factors(n: int=1,
                                    n_factors: int=1,
                                    eps: float=0.01,
                                   ) -> np.ndarray:
    """Generate a random correlation matrix (symmetric positive semi-definite).
    For that :
    1) Generate n_factors random factors for each of the n assets (similar to 
    assuming the n assets are driven by n_factors independant sources of noise)
    2) Compute the covariance of those n assets
    3) Add a small noise of magnitude eps to the covariance eigenvalues to ensure full-rank 
    """
    
    factor_matrix = rd.randn(n, n_factors)
    cov_factor_matrix = np.dot(factor_matrix, factor_matrix.T)
    # scramble the eigenvalues of S a little 
    # if the goal is to generate full-rank correlation matrix
    cov_factor_matrix += np.diag(1*rd.rand(n))
    factor_vols = np.diag(1/np.sqrt(np.diag(cov_factor_matrix)))
    return np.dot(factor_vols, np.dot(cov_factor_matrix, factor_vols))

def single_corr_matrix(n: int=1, 
                       rho: float=1.0):
    """
    Returns a correlation matrix where all variable have same correlation rho
    """
    if rho < 0. or rho >1.:
        raise NameError('Rho is a correlation and therefore must be between 0 and 1')
        
    return np.ones(n)*rho+np.diag(np.ones((n,)))*(1-rho)

def shrink_corr_matrix(corr_matrix: np.ndarray, 
                       shrink_factor: float=0.0):
    """Shrink corr_matrix towards identity
    """
    n = corr_matrix.shape[0]
    return (1-shrink_factor)*corr+shrink_factor*np.eye(n)