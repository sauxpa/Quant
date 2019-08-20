#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from .ou_fitting_base import OU_fitter

class OU_jump_Laplace_fitter(OU_fitter):
    """Fit a OU jump diffusion with constant intensity m and 
    Laplace distribution of jump size with parameter gamma:
    dX_t = mean_reversion*(long_term-X_t)*dt + vol*dW_t + dJ_t
    """
    def __init__(self,
                 df: pd.DataFrame=pd.DataFrame(),
                 std_dev_weight: float=1.0,
                 theta_init: list=[None, None, None, None, None],
                 theta_init_mode: str='random',
                 init_min: list=[1.0, -1.0, 0.1, 0.05, 0.05],
                 init_max: list=[10.0, 1.0, 1.0, 1.0, 5.0],
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
        
        self._bounds = ((None, None), (None, None), (0, None), (0, None), (0, None))
    
    def char_func_theoretical(self, u: float, theta:list ) -> np.complex128:
        """Theoretical characteristic function u -> E[exp(juX_t]] where
        X_t follows a OU diffusion with jumps.
        dX_t = mean_reversion*(long_term-X_t)*dt + vol*dW_t + dJ_t
        Jumps arrive with constant intensity m and size with Laplace distribution of parameter gamma
        """
        mean_reversion, long_term, vol, gamma, m = theta
        return np.exp(u*long_term*1j-(u**2)*(vol**2)/(4*mean_reversion))*(gamma**2/(gamma**2+u**2))**(m/(2*mean_reversion))
    
class OU_jump_only_Laplace_fitter(OU_fitter):
    """Fit a OU jump diffusion with constant intensity m and 
    Laplace distribution of jump size with parameter gamma:
    dX_t = mean_reversion*(long_term-X_t)*dt + vol*dW_t + dJ_t.
    mean_reversion, long_term and vol are fixed, only jump parameters are optimized.
    """
    def __init__(self,
                 df: pd.DataFrame=pd.DataFrame(),
                 std_dev_weight: float=1.0,
                 theta_diffusion: list=[None, None, None],
                 theta_init: list=[None, None],
                 theta_init_mode: str='random',
                 init_min: list=[0.05, 0.05],
                 init_max: list=[1.0, 5.0],
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

        self._bounds = ((0, None), (0, None))
        
        # list of [mean_reversion, long_term, vol]
        self._theta_diffusion = theta_diffusion
    
    @property
    def theta_diffusion(self) -> list:
        return self._theta_diffusion
    @theta_diffusion.setter
    def theta_diffusion(self, new_theta_diffusion) -> None:
        self._theta_diffusion = new_theta_diffusion
        
    def char_func_theoretical(self, u: float, theta:list ) -> np.complex128:
        """Theoretical characteristic function u -> E[exp(juX_t]] where
        X_t follows a OU diffusion with jumps.
        dX_t = mean_reversion*(long_term-X_t)*dt + vol*dW_t + dJ_t
        Jumps arrive with constant intensity m and size with Laplace distribution of parameter gamma
        """
        mean_reversion, long_term, vol = self.theta_diffusion 
        gamma, m = theta
        return np.exp(u*long_term*1j-(u**2)*(vol**2)/(4*mean_reversion))*(gamma**2/(gamma**2+u**2))**(m/(2*mean_reversion))
    
class OU_jump_Laplace_fixed_mr_fitter(OU_fitter):
    """Fit a OU jump diffusion with constant intensity m and 
    Laplace distribution of jump size with parameter gamma:
    dX_t = mean_reversion*(long_term-X_t)*dt + vol*dW_t + dJ_t.
    mean_reversion is fixed, the other parameters are optimized.
    """
    def __init__(self,
                 df: pd.DataFrame=pd.DataFrame(),
                 std_dev_weight: float=1.0,
                 theta_init: list=[None, None, None, None],
                 theta_init_mode: str='random',
                 init_min: list=[-1.0, 0.1, 0.05, 0.05],
                 init_max: list=[1.0, 1.0, 1.0, 5.0],
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

        self._bounds = ((None, None), (0, None),(0, None), (0, None))
        
    def char_func_theoretical(self, u: float, theta:list ) -> np.complex128:
        """Theoretical characteristic function u -> E[exp(juX_t]] where
        X_t follows a OU diffusion with jumps.
        dX_t = mean_reversion*(long_term-X_t)*dt + vol*dW_t + dJ_t
        Jumps arrive with constant intensity m and size with Laplace distribution of parameter gamma
        """
        mean_reversion = self.theta_regression()[0]
        long_term, vol, gamma, m = theta
        return np.exp(u*long_term*1j-(u**2)*(vol**2)/(4*mean_reversion))*(gamma**2/(gamma**2+u**2))**(m/(2*mean_reversion))
    
class OU_jump_Laplace_fixed_vol_fitter(OU_fitter):
    """Fit a OU jump diffusion with constant intensity m and 
    Laplace distribution of jump size with parameter gamma:
    dX_t = mean_reversion*(long_term-X_t)*dt + vol*dW_t + dJ_t.
    vol is fixed, the other parameters are optimized.
    """
    def __init__(self,
                 df: pd.DataFrame=pd.DataFrame(),
                 std_dev_weight: float=1.0,
                 theta_init: list=[None, None, None, None],
                 theta_init_mode: str='random',
                 init_min: list=[1.0, -1.0, 0.05, 0.05],
                 init_max: list=[10.0, 1.0, 1.0, 5.0],
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

        self._bounds = ((0, None), (None, None),(0, None), (0, None))
        
    def char_func_theoretical(self, u: float, theta:list ) -> np.complex128:
        """Theoretical characteristic function u -> E[exp(juX_t]] where
        X_t follows a OU diffusion with jumps.
        dX_t = mean_reversion*(long_term-X_t)*dt + vol*dW_t + dJ_t
        Jumps arrive with constant intensity m and size with Laplace distribution of parameter gamma
        """
        vol = self.vol_estimate()
        mean_reversion, long_term, gamma, m = theta
        return np.exp(u*long_term*1j-(u**2)*(vol**2)/(4*mean_reversion))*(gamma**2/(gamma**2+u**2))**(m/(2*mean_reversion))    