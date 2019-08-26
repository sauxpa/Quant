#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from .ou_fitting_base import OU_fitter

class OU_no_jump_fitter(OU_fitter):
    """Fit a continuous OU diffusion:
    dX_t = mean_reversion*(long_term-X_t)*dt + vol*dW_t
    """
    def __init__(self,
                 df: pd.DataFrame=pd.DataFrame(),
                 std_dev_weight: float=1.0,
                 theta_init: list=[None, None, None],
                 theta_init_mode: str='random',
                 init_min: list=[1.0, -1.0, 0.1],
                 init_max: list=[10.0, 1.0, 1.0],
                 integration_mode: str='hermite',
                 hist_vol_mode: str='mad',
                 n_quadrature: int=10,
                ) -> None:
    
        super().__init__(
            df=df,
            std_dev_weight=std_dev_weight,
            theta_init=theta_init,
            theta_init_mode=theta_init_mode,
            init_min=init_min,
            init_max=init_max,
            integration_mode=integration_mode,
            hist_vol_mode=hist_vol_mode,
            n_quadrature=n_quadrature,            
        )
        
        self._bounds = ((None, None), (None, None), (0, None))
        
    def char_func_theoretical(self, u: float, theta:list ) -> np.complex128:
        """Theoretical characteristic function u -> E[exp(juX_t]] where
        X_t follows a OU diffusion:
        dX_t = mean_reversion*(long_term-X_t)*dt + vol*dW_t
        """
        mean_reversion, long_term, vol = theta
        return np.exp(u*long_term*1j-(u**2)*(vol**2)/(4*mean_reversion))
    
class OU_no_jump_vol_fixed_fitter(OU_fitter):
    """Fit a continuous OU diffusion:
    dX_t = mean_reversion*(long_term-X_t)*dt + vol*dW_t.
    Optimize mean_reversion and long_term only, not the vol.
    The stationary form of the optimization problem for all three
    parameters is not identifiable (it depends on the ratio of mean_reversion
    and vol**2, not on the individual parameters).
    """
    def __init__(self,
                 df: pd.DataFrame=pd.DataFrame(),
                 std_dev_weight: float=1.0,
                 theta_init: list=[None, None],
                 theta_init_mode: str='random',
                 init_min: list=[1.0, -1.0],
                 init_max: list=[10.0, 1.0],
                 integration_mode: str='hermite',
                 hist_vol_mode: str='mad',
                 n_quadrature: int=10,
                ) -> None:
    
        super().__init__(
            df=df,
            std_dev_weight=std_dev_weight,
            theta_init=theta_init,
            theta_init_mode=theta_init_mode,
            init_min=init_min,
            init_max=init_max,
            integration_mode=integration_mode,
            hist_vol_mode=hist_vol_mode,
            n_quadrature=n_quadrature,            
        )
        
        self._bounds = ((None, None), (None, None))
    
    def char_func_theoretical(self, u: float, theta:list ) -> np.complex128:
        """Theoretical characteristic function u -> E[exp(juX_t]] where
        X_t follows a OU diffusion:
        dX_t = mean_reversion*(long_term-X_t)*dt + vol*dW_t
        """
        mean_reversion, long_term = theta
        return np.exp(u*long_term*1j-(u**2)*(self.vol_estimate()**2)/(4*mean_reversion))