#!/usr/bin/env python
# coding: utf-8

import abc
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import roots_hermitenorm, erfinv
from scipy.stats import iqr
from functools import lru_cache

class OU_fitter(abc.ABC):
    """Estimate OU parameters (mean-reversion, long-term mean, volatility, potentially jump parameters) by the generalized method of moments.
    Generic class, the theoretical formula for the characteristic function of OU has to be implemented in child classes as its form varies depending on the assumptions (with or without jumps, jump size distribution...)
    """
    def __init__(self, 
                 df: pd.DataFrame=pd.DataFrame(),
                 std_dev_weight: float=1.0,
                 theta_init: list=[],
                 theta_init_mode: str='random',
                 init_min: list=[],
                 init_max: list=[],
                 integration_mode: str='hermite',
                 hist_vol_mode: str='robust',
                 n_quadrature: int=10,
                 regularization: float=0.0
                ) -> None:
        # empirical data
        self._df = df
        
        # weight function used to aggregated the characteric function loss is 
        # 1/sqrt(2*pi*std_dev_weight**2)*exp(-u**2/(2*std_dev_weight**2))
        self._std_dev_weight = std_dev_weight
        
        # starting point for the estimation of the set of OU parameters
        # if a component is None, it'll be drawn at random
        self._theta_init = theta_init
        
        # either initialize theta by randomly drawing values between given bounds
        # or start from values obtained by a quick regression.
        # theta_init can also be overriden to arbitrary values directly
        self._theta_init_mode = theta_init_mode
        
        # array of fitter parameters
        self._theta = theta_init
        
        # list of boundaries for random initialization of parameters
        self._init_min = init_min
        self._init_max = init_max
        
        # bounds for constrained optimization
        self._bounds = []
        
        # integration mode : scipy QUADPACK quad or Hermite-Gauss quadrature
        self._integration_mode = integration_mode
        
        # estimation method for historic vol : 'standard' computes the variance
        # of the time scaled increments, 'robust' uses interquantile range (more
        # robust to outliers, in particular robust to jumps)
        self._hist_vol_mode = hist_vol_mode
        
        # in Hermite-Gauss mode, number of points for quadrature
        self._n_quadrature = n_quadrature
        
        # L2 regularization strength for theta
        self._regularization = regularization
        
    @property
    def df(self) -> pd.DataFrame:
        return self._df
    @df.setter
    def df(self, new_df) -> None:
        self.vol_estimate.cache_clear()
        self.theta_regression.cache_clear()
        self._df = new_df
    
    @property
    def std_dev_weight(self) -> float:
        return self._std_dev_weight
    @std_dev_weight.setter
    def std_dev_weight(self, new_std_dev_weight) -> None:
        self._std_dev_weight = new_std_dev_weight

    @property
    def theta_init_mode(self) -> float:
        return self._theta_init_mode
    @theta_init_mode.setter
    def theta_init_mode(self, new_theta_init_mode) -> None:
        self._theta_init_mode = new_theta_init_mode
        
    @property
    def init_min(self) -> list:
        """Lower bound for random initialization the set of OU parameters
        """
        return self._init_min
    @init_min.setter
    def init_min(self, new_init_min) -> None:
        self._init_min = new_init_min
    
    @property
    def init_max(self) -> list:
        """Upper bound for random initialization the set of OU parameters
        """
        return self._init_max
    @init_max.setter
    def init_max(self, new_init_max) -> None:
        self._init_max = new_init_max
        
    @property
    def integration_mode(self) -> str:
        return self._integration_mode
    @integration_mode.setter
    def integration_mode(self, new_integration_mode: str) -> None:
        self._integration_mode = new_integration_mode
        
    @property
    def hist_vol_mode(self) -> str:
        return self._hist_vol_mode
    @hist_vol_mode.setter
    def hist_vol_mode(self, new_hist_vol_mode: str) -> None:
        self.vol_estimate.cache_clear()
        self.theta_regression.cache_clear()
        self._hist_vol_mode = new_hist_vol_mode
        
    @property
    def n_quadrature(self) -> int:
        return self._n_quadrature
    @n_quadrature.setter
    def n_quadrature(self, new_n_quadrature) -> None:
        self.hermite_gauss_info.cache_clear()
        self._n_quadrature = new_n_quadrature

    @property
    def regularization(self) -> float:
        return self._regularization
    @regularization.setter
    def regularization(self, new_regularization) -> None:
        self._regularization = new_regularization

    @lru_cache(maxsize=None)
    def hermite_gauss_info(self):
        return roots_hermitenorm(self.n_quadrature)
        
    @property
    def theta_init(self) -> list:
        if self.theta_init_mode == 'random':
            self._theta_init = [p if p else np.random.uniform(self.init_min[i], self.init_max[i]) for i, p in enumerate(self._theta_init)]
            return self._theta_init
        elif self.theta_init_mode == 'regression':
            self._theta_init = self.theta_regression()[:len(self._theta_init)]
            return self._theta_init
        elif self.theta_init_mode == 'manual':
            return self._theta_init
        else:
            raise NameError('Unknown initialization mode : {}'.format(self.theta_init_mode))
    @theta_init.setter
    def theta_init(self, new_theta_init) -> None:
        if self.theta_init_mode == 'manual':
            self._theta_init = new_theta_init
        else:
            raise Exception('Cannot initialize theta manually in {} mode'.format(self.theta_init_mode))
            
    @property
    def theta(self) -> list:
        return self._theta
    
    @property
    def df_inc(self) -> pd.DataFrame:
        return (self.df-self.df.shift(1)).iloc[1:]

    @property
    def df_scaled(self) -> pd.DataFrame:
        """Scale series by the time steps
        """
        index = np.array(self.df.index)
        inc_index = index[1:]-index[:-1]
        inc_index_df = pd.DataFrame(inc_index,
                                    columns=self.df.columns, 
                                    index=self.df.iloc[:-1].index,
                                   )
        return self.df.iloc[:-1]*inc_index_df
    
    @property
    def df_inc_normalized(self) -> pd.DataFrame:
        """Normalized increments i.e scale by the square root
        of the time steps
        """
        index = np.array(self.df.index)
        inc_index_sqrt = np.sqrt(index[1:]-index[:-1])
        inc_index_sqrt_df = pd.DataFrame(inc_index_sqrt,
                                         columns=self.df_inc.columns, 
                                         index=self.df_inc.index
                                        )
        return self.df_inc/inc_index_sqrt_df
    
    @lru_cache(maxsize=None)
    def vol_estimate(self):
        """Estimate volatility of the driving noise by computing the standard
        deviation of the rescaled increments
        """
        if self.hist_vol_mode == 'std':
            return self.df_inc_normalized.std()[0]
        elif self.hist_vol_mode == 'robust':
            return iqr(self.df_inc_normalized)/(2*np.sqrt(2)*erfinv(0.5))
        else:
            raise NameError('Unknown historical vol estimation mode : {}'.format(self.hist_vol_mode))

    @lru_cache(maxsize=None)
    def theta_regression(self) -> list:
        """Estimation of OU parameters via regression:
            * mean-reversion : regression of negative returns on the values of the process
            * long_term : mean value of the process
            * vol : time-scaled standard deviation of returns 
        """
        return [
                -np.linalg.lstsq(self.df_scaled, self.df_inc, rcond=None)[0][0][0],
                self.df.mean()[0],
                self.vol_estimate(),
               ]
    
    @abc.abstractmethod
    def  char_func_theoretical(self, u: float, theta: list) -> np.complex128:
        """To be instanciated in child classes
        """
        pass
    
    def char_func_empirical(self,
                            u: float,
                           ) -> np.complex128:
        """Empirical characteristic function u -> 1/n*sum_{i=1}^n(exp(juX_i)) where X_i are data samples
        """
        return np.exp(self.df*u*1j).mean()[0].astype(np.complex128)
    
    def char_func_loss(self,
                       u: float,
                       theta: list,
                      ) -> float:
        return np.abs(self.char_func_theoretical(u, theta)-self.char_func_empirical(u)) ** 2
    
    def weight_func(self, u: float) -> float:
        return 1/(np.sqrt(2*np.pi)*self.std_dev_weight)*np.exp(-u**2/(2*self.std_dev_weight**2))
    
    def objective_func(self, theta: list ) -> float:
        if self.integration_mode == 'quad':
            func = lambda u: self.weight_func(u)*self.char_func_loss(u, theta)
            ret = quad(func, -np.inf, np.inf)[0]
            if self.regularization > 0:
                ret += self.regularization*np.linalg.norm(theta)
            return ret
        elif self.integration_mode == 'hermite':
            knots, weights = self.hermite_gauss_info()
            ret = np.dot(list(map(lambda u: self.char_func_loss(self.std_dev_weight*u, theta), knots)), weights) 
            if self.regularization > 0:
                ret += self.regularization*np.linalg.norm(theta)
            return ret
        else:
            raise NameError('Unknown integration mode : {}'.format(self.integration_mode))
        
    def fit(self) -> None:
        fit_info = minimize(self.objective_func, 
                            self.theta_init,
                            bounds=self._bounds,
                            method='L-BFGS-B',
                           )
        if fit_info.success:
            self._theta = fit_info.x
            return fit_info
        else:
            raise Exception('Failed calibration...')