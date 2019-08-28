#!/usr/bin/env python
# coding: utf-8

import numpy as np
import abc
from volmodels import Vol_model, ONE_BP

# Displaced lognormal
# Pure local volatility model here
class SLN_adjustable_backbone_local_vol(Vol_model):
    """Shifted lognormal model with adjustable backbone as developped 
    in Andersen and Piterbarg in (16.5):
    dF_t = sigma*(shift*F_t+(mixing-shift)*F_0+(1-m)*L)*dW_t
    """
    def __init__(self, 
                 sigma: float=1.0,
                 shift: float=0.0,
                 mixing: float=0.0,
                 L: float=0.0,
                 f=None,
                 T_expiry: float=1.0,
                 vol_type=None,
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50
                ) -> None:
        
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
        
        self._sigma = sigma
        self._shift = shift
        self._mixing = mixing
        self._L = L
    
    @property
    def sigma(self) -> float:
        return self._sigma
    @sigma.setter
    def sigma(self, new_sigma: float) -> None:
        self._sigma = new_sigma
    
    @property
    def shift(self) -> float:
        return self._shift
    @shift.setter
    def shift(self, new_shift: float) -> None:
        self._shift = new_shift
    
    @property
    def mixing(self) -> float:
        return self._mixing
    @mixing.setter
    def mixing(self, new_mixing: float) -> None:
        self._mixing = new_mixing
        
    @property
    def L(self) -> float:
        return self._L
    @L.setter
    def L(self, new_L: float) -> None:
        self._L = new_L
        
    def __str__(self) -> str:
        return r'$\sigma$={:.2%}, shift={:.2f}, mixing={:.2f}, L={:.2f}, f={:.2%}'        .format(self.sigma, self.shift, self.mixing, self.L, self.f)
    
    @abc.abstractmethod
    def smile_func(self, K):
        pass

class SLN_adjustable_backbone_local_vol_LN(SLN_adjustable_backbone_local_vol):
    """Shifted lognormal model with adjustable backbone in lognormal quoting
    """
    def __init__(self, 
                 sigma: float=1.0,
                 shift: float=0.0,
                 mixing: float=0.0,
                 L: float=0.0,
                 f=None, 
                 T_expiry: float=1.0,
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50,
                ) -> None:
        super().__init__(sigma=sigma, 
                         shift=shift,
                         mixing=mixing,
                         L=L,
                         f=f, 
                         T_expiry=T_expiry,
                         vol_type='LN',
                         logmoneyness_lo=logmoneyness_lo,
                         logmoneyness_hi=logmoneyness_hi,
                         K_lo=K_lo,
                         K_hi=K_hi,
                         strike_type=strike_type,
                         n_strikes=n_strikes,
                        )
        
    @property
    def model_name(self) -> str:
        return 'SLN_adjustable_backbone_local_vol_LN'
    
    @property
    def ATM_LN(self) -> float:
        local_vol_f = self.mixing*self.f+(1-self.mixing)*self.L
        return self.sigma*(self.mixing*self.f+(1-self.mixing)*self.L)/self.f    *(1+1/24*((-(self.shift/local_vol_f)**2+1/(self.f**2))*(self.sigma**2)*(local_vol_f**2))      *self.T_expiry)
        
    @property
    def ATM(self) -> float:
        return self.ATM_LN
    
    def smile_func(self, K: float) -> float:
        """Implied vol is the harmonic average of local vol on the path F_0 -> K
        """
        if np.abs(K-self.f)<ONE_BP/100:
            return self.ATM
        else:
            log_moneyness = np.log(self.f/K)
            f_avg = np.sqrt(self.f*K)
            int_inv_local_vol = np.log(                                       (self.mixing*self.f+(1-self.mixing)*self.L)/                                       (self.shift*K+(self.mixing-self.shift)*self.f                                        +(1-self.mixing)*self.L))
            local_vol_f_avg = self.shift*f_avg            +(self.mixing-self.shift)*self.f            +(1-self.mixing)*self.L
            gamma_1 = self.shift/local_vol_f_avg 
            
            return self.sigma*self.shift*log_moneyness/int_inv_local_vol        *(1+((1/(f_avg**2)-gamma_1**2)/24*(self.sigma**2)*(local_vol_f_avg**2))*self.T_expiry)
        
class SLN_adjustable_backbone_local_vol_N(SLN_adjustable_backbone_local_vol):
    """Shifted lognormal model with adjustable backbone in normal quoting
    """
    def __init__(self, 
                 sigma: float=1.0,
                 shift: float=0.0,
                 mixing: float=0.0,
                 L: float=0.0,
                 f=None, 
                 T_expiry: float=1.0,
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50,
                ) -> None:
        super().__init__(sigma=sigma, 
                         shift=shift,
                         mixing=mixing,
                         L=L,
                         f=f, 
                         T_expiry=T_expiry,
                         vol_type='N',
                         logmoneyness_lo=logmoneyness_lo,
                         logmoneyness_hi=logmoneyness_hi,
                         K_lo=K_lo,
                         K_hi=K_hi,
                         strike_type=strike_type,
                         n_strikes=n_strikes,
                        )
        
    @property
    def model_name(self) -> str:
        return 'SLN_adjustable_backbone_local_vol_N'
    
    @property
    def ATM_N(self) -> float:
        local_vol_f = self.mixing*self.f+(1-self.mixing)*self.L
        return self.sigma*local_vol_f*(1+1/24*((-(self.shift/local_vol_f)**2)*(self.sigma**2)*(local_vol_f**2))*self.T_expiry)
        
    @property
    def ATM(self) -> float:
        return self.ATM_N
    
    def smile_func(self, K: float) -> float:
        """Implied vol is the harmonic average of local vol on the path F_0 -> K
        """
        if np.abs(K-self.f)<ONE_BP/100:
            return self.ATM
        else:
            f_avg = np.sqrt(self.f*K)
            int_inv_local_vol = np.log((self.mixing*self.f + (1-self.mixing)*self.L)/(self.shift*K + (self.mixing-self.shift)*self.f + (1-self.mixing)*self.L))
            local_vol_f_avg = self.shift*f_avg + (self.mixing-self.shift)*self.f + (1-self.mixing)*self.L
            gamma_1 = self.shift/local_vol_f_avg 
            
            return self.sigma*self.shift*(self.f-K)/int_inv_local_vol*(1 + (-gamma_1**2/24*(self.sigma**2)*(local_vol_f_avg**2))*self.T_expiry)        
        
# Displaced lognormal with stochastic volatility
class SLN_adjustable_backbone(Vol_model):
    """Shifted lognormal model with adjustable backbone
    dF_t = sigma_t*(shift*F_t+(mixing-shift)*F_0+(1-m)*L)*dW_t
    dsigma_t = vov*sigma_t*dB_t
    d<B,W>_t = rho*dt
    """
    def __init__(self, 
                 sigma_0: float=1.0,
                 vov: float=1.0,
                 rho: float=0.0,
                 shift: float=0.0,
                 mixing: float=0.0,
                 L: float=0.0,
                 f=None,
                 T_expiry: float=1.0,
                 vol_type=None,
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50
                ) -> None:
        
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
        
        self._sigma_0 = sigma_0
        self._vov = vov
        self._rho = rho
        self._shift = shift
        self._mixing = mixing
        self._L = L
    
    @property
    def sigma_0(self) -> float:
        return self._sigma_0
    @sigma_0.setter
    def sigma_0(self, new_sigma_0: float) -> None:
        self._sigma_0 = new_sigma_0
        
    @property
    def vov(self) -> float:
        return self._vov
    @vov.setter
    def vov(self, new_vov: float) -> None:
        self._vov = new_vov
    
    @property
    def rho(self) -> float:
        return self._rho
    @rho.setter
    def rho(self, new_rho: float) -> None:
        self._rho = new_rho
        
    @property
    def shift(self) -> float:
        return self._shift
    @shift.setter
    def shift(self, new_shift: float) -> None:
        self._shift = new_shift
    
    @property
    def mixing(self) -> float:
        return self._mixing
    @mixing.setter
    def mixing(self, new_mixing: float) -> None:
        self._mixing = new_mixing
        
    @property
    def L(self) -> float:
        return self._L
    @L.setter
    def L(self, new_L: float) -> None:
        self._L = new_L
        
    def __str__(self) -> str:
        return r'$\sigma_0$={:.2%}, vov={:.0%}, $\rho=${:.0%}, shift={:.2f}, mixing={:.2f}, L={:.2f}, f={:.2%}'.format(self.sigma_0, self.vov, self.rho, self.shift, self.mixing, self.L, self.f)
    
    @abc.abstractmethod
    def smile_func(self, K):
        pass

class SLN_adjustable_backbone_LN(SLN_adjustable_backbone):
    """Shifted lognormal model with adjustable backbone in lognormal quoting
    """
    def __init__(self, 
                 sigma_0: float=1.0,
                 vov: float=1.0,
                 rho: float=0.0,
                 shift: float=0.0,
                 mixing: float=0.0,
                 L: float=0.0,
                 f=None, 
                 T_expiry: float=1.0,
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50,
                ) -> None:
        super().__init__(sigma_0=sigma_0,
                         vov=vov,
                         rho=rho,
                         shift=shift,
                         mixing=mixing,
                         L=L,
                         f=f, 
                         T_expiry=T_expiry,
                         vol_type='LN',
                         logmoneyness_lo=logmoneyness_lo,
                         logmoneyness_hi=logmoneyness_hi,
                         K_lo=K_lo,
                         K_hi=K_hi,
                         strike_type=strike_type,
                         n_strikes=n_strikes,
                        )
        
    @property
    def model_name(self) -> str:
        return 'SLN_adjustable_backbone_LN'
    
    @property
    def ATM_LN(self) -> float:
        local_vol_f = self.mixing*self.f+(1-self.mixing)*self.L
        gamma_1 = self.shift/local_vol_f
        return self.sigma_0/self.f*local_vol_f*(1 + ((-gamma_1**2 + 1/(self.f**2))/24*self.sigma_0**2*local_vol_f**2 + 0.25*self.rho*self.vov*self.sigma_0*gamma_1*local_vol_f+(2-3*self.rho**2)/24*self.vov**2)*self.T_expiry)
        
    @property
    def ATM(self) -> float:
        return self.ATM_LN
    
    def smile_func(self, K: float) -> float:
        """Implied vol is the harmonic average of local vol on the path F_0 -> K
        """
        if np.abs(K-self.f)<ONE_BP/100:
            return self.ATM
        else:
            log_moneyness = np.log(self.f/K)
            f_avg = np.sqrt(self.f*K)
            
            int_inv_local_vol = 1/self.shift*np.log((self.mixing*self.f + (1 - self.mixing)*self.L)/(self.shift*K + (self.mixing-self.shift)*self.f + (1 - self.mixing)*self.L))
            
            local_vol_f_avg = self.shift*f_avg + (self.mixing - self.shift)*self.f + (1 - self.mixing)*self.L
            
            gamma_1 = self.shift/local_vol_f_avg 
            
            zeta = self.vov/self.sigma_0*(self.f-K)/local_vol_f_avg
            q_zeta = 1-2*self.rho*zeta+zeta**2
            x_zeta = np.log((np.sqrt(q_zeta)-self.rho+zeta)/(1-self.rho))
            
            return self.sigma_0*log_moneyness/int_inv_local_vol*zeta/x_zeta*(1 + ((-gamma_1**2 + 1/(f_avg**2))/24*self.sigma_0**2*local_vol_f_avg**2 + 0.25*self.rho*self.vov*self.sigma_0*gamma_1*local_vol_f_avg + (2-3*self.rho**2)/24*self.vov**2)*self.T_expiry)
        
class SLN_adjustable_backbone_N(SLN_adjustable_backbone):
    """Shifted lognormal model with adjustable backbone in normal quoting
    """
    def __init__(self, 
                 sigma_0: float=1.0,
                 vov: float=1.0,
                 rho: float=0.0,
                 shift: float=0.0,
                 mixing: float=0.0,
                 L: float=0.0,
                 f=None, 
                 T_expiry: float=1.0,
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50,
                ) -> None:
        super().__init__(sigma_0=sigma_0,
                         vov=vov,
                         rho=rho,
                         shift=shift,
                         mixing=mixing,
                         L=L,
                         f=f, 
                         T_expiry=T_expiry,
                         vol_type='N',
                         logmoneyness_lo=logmoneyness_lo,
                         logmoneyness_hi=logmoneyness_hi,
                         K_lo=K_lo,
                         K_hi=K_hi,
                         strike_type=strike_type,
                         n_strikes=n_strikes,
                        )
        
    @property
    def model_name(self) -> str:
        return 'SLN_adjustable_backbone_N'
    
    @property
    def ATM_N(self) -> float:
        local_vol_f = self.mixing*self.f+(1-self.mixing)*self.L
        gamma_1 = self.shift/local_vol_f
        return self.sigma_0*local_vol_f*(1 + ((-gamma_1**2)/24*self.sigma_0**2*local_vol_f**2 + 0.25*self.rho*self.vov*self.sigma_0*gamma_1*local_vol_f+(2-3*self.rho**2)/24*self.vov**2)*self.T_expiry)
        
    @property
    def ATM(self) -> float:
        return self.ATM_N
    
    def smile_func(self, K: float) -> float:
        """Implied vol is the harmonic average of local vol on the path F_0 -> K
        """
        if np.abs(K-self.f)<ONE_BP/100:
            return self.ATM
        else:
            f_avg = np.sqrt(self.f*K)
            
            int_inv_local_vol = 1/self.shift*np.log((self.mixing*self.f + (1 - self.mixing)*self.L)/(self.shift*K + (self.mixing-self.shift)*self.f + (1 - self.mixing)*self.L))
            
            local_vol_f_avg = self.shift*f_avg + (self.mixing - self.shift)*self.f + (1 - self.mixing)*self.L
            
            gamma_1 = self.shift/local_vol_f_avg 
            
            zeta = self.vov/self.sigma_0*(self.f-K)/local_vol_f_avg
            q_zeta = 1-2*self.rho*zeta+zeta**2
            x_zeta = np.log((np.sqrt(q_zeta)-self.rho+zeta)/(1-self.rho))
            
            return self.sigma_0*(self.f-K)/int_inv_local_vol*zeta/x_zeta*(1 + ((-gamma_1**2)/24*self.sigma_0**2*local_vol_f_avg**2 + 0.25*self.rho*self.vov*self.sigma_0*gamma_1*local_vol_f_avg + (2-3*self.rho**2)/24*self.vov**2)*self.T_expiry)              