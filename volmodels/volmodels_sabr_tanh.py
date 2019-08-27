#!/usr/bin/env python
# coding: utf-8

import numpy as np
import abc
from volmodels import Vol_model, ONE_BP

# Modified SABR with tanh local vol
# In regular SABR, beta controls the backbone shape. Standard market practices are to have beta close to 1 in low rates environment (lognormal SABR) and closer to 0 when rates hike (normal SABR). 
# tanh(x) has a unit slope close to zero and flattens for larger x, so using it as a local vol function allows to dynamically mirror the remarking of beta as rates move. Moreover, it yields closed-form vol expansion in Hagan's formula
class SABR_tanh_base_model(Vol_model):
    """Generic class for SABR_tanh model
    dX_t = s_t*C(X_t)*dW_t
    ds_t = vov*s_t*dB_t
    d<W,B>_t = rho*dt
    C(x) = tanh(x/l)
    """
    def __init__(self, 
                 l: float=1.0,
                 vov: float=1.0,
                 rho: float=0.0,
                 sigma_0=None,
                 f=None,
                 T_expiry: float=1.0,
                 vol_type=None,
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50,
                ) -> None:
        """Implementation of SABR implied vol and approximated price formula.
        l, vov and rho are marked.
        """
        super().__init__(f=f, 
                         T_expiry=T_expiry,
                         vol_type=vol_type,
                         logmoneyness_lo=logmoneyness_lo,
                         logmoneyness_hi=logmoneyness_hi,
                         K_lo=K_lo,
                         K_hi=K_hi,
                         strike_type=strike_type,
                         n_strikes=n_strikes
                        )
        self._l = l
        self._vov = vov
        self._rho = rho
        self._sigma_0 = sigma_0

    @property
    def l(self):
        return self._l
    @l.setter
    def l(self, new_l):
        self._l = new_l
    
    @property
    def rho(self):
        return self._rho
    @rho.setter
    def rho(self, new_rho):
        self._rho = new_rho
        
    @property
    def vov(self):
        return self._vov
    @vov.setter
    def vov(self, new_vov):
        self._vov = new_vov
    
    @property
    def sigma_0(self):
        return self._sigma_0
    @sigma_0.setter
    def sigma_0(self, new_sigma_0):
        self._sigma_0 = new_sigma_0
        
    def __str__(self):
        return r'l={:.2f}, vov={:.0%}, $\rho$={:.0%}, ATM={:.2%}, f={:.2%}, $\sigma_0$={:.2%}'        .format(self.l, self.vov, self.rho, self.ATM, self.f, self.sigma_0)
    
    def local_vol_inv_int(self, x):
        """Closed form for the primitive of 1/(tanh((x+s)/l))
        """
        return self.l*np.log(np.sinh(x/self.l))
    
    @abc.abstractmethod
    def smile_func(self, K):
        pass

class SABR_tanh_LN(SABR_tanh_base_model):
    """SABR_tanh model using lognormal implied vol quoting
    """
    def __init__(self, 
                 l: float=1.0,
                 vov: float=1.0,
                 rho: float=0.0,
                 sigma_0=None,
                 f=None,
                 T_expiry: float=1.0,
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50,
                ) -> None:
        super().__init__(l=l, 
                         vov=vov,
                         rho=rho,
                         sigma_0=sigma_0,
                         f=f,
                         T_expiry=T_expiry,
                         vol_type='LN',
                         logmoneyness_lo=logmoneyness_lo,
                         logmoneyness_hi=logmoneyness_hi,
                         K_lo=K_lo,
                         K_hi=K_hi,
                         strike_type=strike_type,
                         n_strikes=n_strikes
                        )
    
    @property
    def model_name(self):
        return 'SABR_tanh_LN'
    
    @property
    def ATM_LN(self):
        f_adj = self.f/self.l
        C_f_adj = np.tanh(f_adj)
        gamma_1 = 1/(self.l*np.cosh(f_adj)**2)/C_f_adj
        gamma_2 = -2/(self.l**2*np.cosh(f_adj)**2)
        return self.sigma_0/self.f*C_f_adj    *(1      +((2*gamma_2-gamma_1**2+1/(self.f**2))/24*self.sigma_0**2*C_f_adj**2       +0.25*self.rho*self.vov*self.sigma_0*gamma_1*C_f_adj       +(2-3*self.rho**2)/24*self.vov**2)      *self.T_expiry)

    @property
    def ATM(self):
        return self.ATM_LN
    
    def smile_func(self, K):
        """Hagan lognormal smile approximation around ATM as written in the
        original 'Managing Smile Risk' paper using generic local vol (A.65)
        K: strike
        """
        if np.abs(K-self.f)<ONE_BP/100:
            return self.ATM
        else:
            log_moneyness = np.log(self.f/K)
            f_avg = np.sqrt(self.f*K)
            f_avg_adj = f_avg/self.l
            C_f_avg_adj = np.tanh(f_avg_adj)
            gamma_1 = 1/(self.l*np.cosh(f_avg_adj)**2)/C_f_avg_adj
            gamma_2 = -2/(self.l**2*np.cosh(f_avg_adj)**2)
            denom = self.local_vol_inv_int(self.f)-self.local_vol_inv_int(K)
            zeta = self.vov/self.sigma_0*(self.f-K)/C_f_avg_adj
            q_zeta = 1-2*self.rho*zeta+zeta**2
            x_zeta = np.log((np.sqrt(q_zeta)-self.rho+zeta)/(1-self.rho))
            return self.sigma_0*log_moneyness/denom*zeta/x_zeta        *(1          +((2*gamma_2-gamma_1**2+1/(f_avg**2))/24*self.sigma_0**2*C_f_avg_adj**2           +0.25*self.rho*self.vov*self.sigma_0*gamma_1*C_f_avg_adj           +(2-3*self.rho**2)/24*self.vov**2)          *self.T_expiry)

class SABR_tanh_N(SABR_tanh_base_model):
    """SABR_tanh model using lognormal implied vol quoting
    """
    def __init__(self, 
                 l: float=1.0,
                 vov: float=1.0,
                 rho: float=0.0,
                 sigma_0=None,
                 f=None,
                 T_expiry: float=1.0,
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50,
                ) -> None:
        super().__init__(l=l,
                         vov=vov,
                         rho=rho,
                         sigma_0=sigma_0,
                         f=f,
                         T_expiry=T_expiry,
                         vol_type='N',
                         logmoneyness_lo=logmoneyness_lo,
                         logmoneyness_hi=logmoneyness_hi,
                         K_lo=K_lo,
                         K_hi=K_hi,
                         strike_type=strike_type,
                         n_strikes=n_strikes
                        )
    
    @property
    def model_name(self):
        return 'SABR_tanh_N'
    
    @property
    def ATM_N(self):
        f_adj = self.f/self.l
        C_f_adj = np.tanh(f_adj)
        gamma_1 = 1/(self.l*np.cosh(f_adj)**2)/C_f_adj
        gamma_2 = -2/(self.l**2*np.cosh(f_adj)**2)
        return self.sigma_0*C_f_adj    *(1      +((2*gamma_2-gamma_1**2)/24*self.sigma_0**2*C_f_adj**2       +0.25*self.rho*self.vov*self.sigma_0*gamma_1*C_f_adj       +(2-3*self.rho**2)/24*self.vov**2)      *self.T_expiry)

    @property
    def ATM(self):
        return self.ATM_N
    
    def smile_func(self, K):
        """Hagan lognormal smile approximation around ATM as written in the
        original 'Managing Smile Risk' paper using generic local vol (A.65)
        K: strike
        """
        if np.abs(K-self.f)<ONE_BP/100:
            return self.ATM
        else:
            log_moneyness = np.log(self.f/K)
            f_avg = np.sqrt(self.f*K)
            f_avg_adj = f_avg/self.l
            C_f_avg_adj = np.tanh(f_avg_adj)
            gamma_1 = 1/(self.l*np.cosh(f_avg_adj)**2)/C_f_avg_adj
            gamma_2 = -2/(self.l**2*np.cosh(f_avg_adj)**2)
            denom = self.local_vol_inv_int(self.f)-self.local_vol_inv_int(K)
            zeta = self.vov/self.sigma_0*(self.f-K)/C_f_avg_adj
            q_zeta = 1-2*self.rho*zeta+zeta**2
            x_zeta = np.log((np.sqrt(q_zeta)-self.rho+zeta)/(1-self.rho))
            return self.sigma_0*(self.f-K)/denom*zeta/x_zeta        *(1          +((2*gamma_2-gamma_1**2)/24*self.sigma_0**2*C_f_avg_adj**2           +0.25*self.rho*self.vov*self.sigma_0*gamma_1*C_f_avg_adj           +(2-3*self.rho**2)/24*self.vov**2)          *self.T_expiry)