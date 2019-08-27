#!/usr/bin/env python
# coding: utf-8

import numpy as np
import abc
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.special import erf
from functools import partial
from functools import lru_cache
from ito_diffusions import SABR
from volmodels import Vol_model, ONE_BP

# ## SABR generic class
# Possible SABR calculations include : 
# * Hagan normal implied vol
# * Hagan lognormal implied
# * Monte Carlo
class SABR_base_model(Vol_model):
    """Generic class for SABR model
    """
    def __init__(self, 
                 beta: float=1.0,
                 vov: float=1.0,
                 rho: float=0.0,
                 ATM=None,
                 sigma_0=None,
                 f=None,
                 T_expiry: float=1.0,
                 vol_type=None,
                 marking_mode: str='ATM',
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50
                ) -> None:
        """Implementation of SABR implied vol and approximated price formula.
        beta, vov and rho are marked.
        One can either mark the ATM vol and the forward and solve for sigma_0,
        or mark sigma_0 directly.
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
        self._beta = beta
        self._vov = vov
        self._rho = rho
        
        # ATM or sigma_0
        self._marking_mode = marking_mode
    
        self._ATM = ATM
        self._sigma_0 = sigma_0

        if self._marking_mode == 'ATM' and self._ATM == None:
            raise NameError( "ATM must be marked in ATM marking mode" )
        elif self._marking_mode == 'sigma_0' and self._sigma_0 == None:
            raise NameError( "sigma_0 must be marked in sigma_0 marking mode" )

    def __str__(self):
        return r'T_expiry={}y, $\beta$={:.2f}, vov={:.2%}, $\rho$={:.2%}, f={:.2%}, $\sigma_0$={:.2%}'    .format(self.T_expiry, self.beta, self.vov, self.rho, self.f, self.sigma_0)   
    
    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self, new_beta):
        self._beta = new_beta
    
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
    def ATM_LN(self):
        if self.vol_type == 'LN':
            return self.ATM
        elif self.vol_type == 'N':
            return self.ATM/self.f
    @property
    def ATM_N(self):
        if self.vol_type == 'N':
            return self.ATM
        elif self.vol_type == 'LN':
            return self.ATM*self.f
        
    @property
    def ATM(self):
        if self.marking_mode == 'ATM':
            return self._ATM
        elif self.marking_mode == 'sigma_0':
            return self.ATM_func(self.sigma_0)
    @ATM.setter
    def ATM(self, new_ATM):
        if self.marking_mode == 'sigma_0':
            raise NameError( "Impossible to mark ATM in sigma_0 marking mode" )
        elif self.marking_mode == 'ATM':
            self._ATM = new_ATM

    @property
    def sigma_0(self):
        if self.marking_mode == 'ATM':
            if self.vol_type == 'LN':
                try:
                    return brentq(lambda sigma_0: self.ATM_func(sigma_0) - self.ATM,                                  self.ATM * (self.f ** (1-self.beta))-0.1,                                  self.ATM * (self.f ** (1-self.beta))+0.1)
                except:
                    return self.ATM * (self.f ** (1-self.beta)) 
            elif self.vol_type == 'N':
                try:
                    return brentq(lambda sigma_0: self.ATM_func(sigma_0) - self.ATM,                                  self.ATM * (self.f ** (-self.beta))-0.1,                                  self.ATM * (self.f ** (-self.beta))+0.1)
                except:
                    return self.ATM * (self.f ** (-self.beta))
        elif self.marking_mode == 'sigma_0':
            return self._sigma_0
    @sigma_0.setter
    def sigma_0(self, new_sigma_0):
        if self.marking_mode == "ATM":
            raise NameError( "Impossible to mark sigma_0 in ATM marking mode" )
        elif self.marking_mode == 'sigma_0':
            self._sigma_0 = new_sigma_0
    
    @property
    def marking_mode(self):
        if self._marking_mode not in ['ATM', 'sigma_0']:
            raise NameError( "Unsupported marking mode : {}".format(self._marking_mode) )
        else:
            return self._marking_mode
    @marking_mode.setter
    def marking_mode(self, new_marking_mode):
        self._marking_mode = new_marking_mode
 
    def ATM_LN_func(self, sigma_0):
        """Reduction of Hagan's lognormal formula when K --> f
        Used to calibrate sigma_0 to marked ATM vol
        """
        return sigma_0/(self.f**(1-self.beta))*    (1+(((1-self.beta)**2)/24*sigma_0**2/(self.f**(2-2*self.beta))     +0.25*self.rho*self.beta*self.vov*sigma_0/(self.f**(1-self.beta))     +(2-3*self.rho**2)/24*self.vov**2)*self.T_expiry)
    
    def ATM_N_func(self, sigma_0):
        """Reduction of Hagan's normal formula when K --> f
        Used to calibrate sigma_0 to marked ATM vol
        """
        return sigma_0 * (self.f**self.beta)*    (1+     ((-self.beta*(2-self.beta)*sigma_0**2)/(24*self.f**(2-2*self.beta))     +0.25*self.rho*self.vov*self.beta*sigma_0/(self.f**(1-self.beta))     +(2-3*self.rho**2)/24*self.vov**2)*self.T_expiry)
        
    def ATM_func(self, sigma_0):
        if self.vol_type == 'LN':
            return self.ATM_LN_func(sigma_0)
        elif self.vol_type == 'N':
            return self.ATM_N_func(sigma_0)
    
    @abc.abstractmethod
    def smile_func(self, K):
        pass

# ### Hagan lognormal (Black) implied vol expansion
class SABR_Hagan_LN(SABR_base_model):
    """SABR model using lognormal implied vol quoting from Hagan's formula
    """
    def __init__(self, 
                 beta: float=1.0,
                 vov: float=1.0,
                 rho: float=0.0,
                 ATM=None,
                 sigma_0=None,
                 f=None,
                 T_expiry: float=1.0,
                 marking_mode: str='ATM',
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50,
                ) -> None:
        super().__init__(beta=beta,
                         vov=vov,
                         rho=rho,
                         ATM=ATM,
                         sigma_0=sigma_0,
                         f=f,
                         T_expiry=T_expiry,
                         vol_type='LN',
                         marking_mode=marking_mode,
                         logmoneyness_lo=logmoneyness_lo,
                         logmoneyness_hi=logmoneyness_hi,
                         K_lo=K_lo,
                         K_hi=K_hi,
                         strike_type=strike_type,
                         n_strikes=n_strikes
                        )
    
    def __str__(self):
        return r'$T_expiry={}y, \beta$={:.2f}, vov={:.2%}, $\rho$={:.2%}, ATM={:.2%}, f={:.2%}, $\sigma_0$={:.2%}'    .format(self.T_expiry, self.beta, self.vov, self.rho, self.ATM, self.f, self.sigma_0)
    
    @property
    def model_name(self):
        return 'SABR_Hagan_LN'
    
    def smile_func(self, K):
        """Hagan lognormal smile approximation around ATM as written in the
        original 'Managing Smile Risk' paper (A.69c)
        K: strike
        """
        if np.abs(K-self.f)<ONE_BP/100:
            return self.ATM
        else:
            log_moneyness = np.log(self.f/K)
            zeta = self.vov/self.sigma_0*(self.f*K)**((1-self.beta)/2)*log_moneyness
            q_zeta = 1-2*self.rho*zeta+zeta**2
            x_zeta = np.log((np.sqrt(q_zeta)-self.rho+zeta)/(1-self.rho))
            f_avg = np.sqrt(self.f*K)
            return self.sigma_0/                                (f_avg**(1-self.beta)*(1                                    +((1-self.beta)**2)/24*log_moneyness**2                                   +((1-self.beta)**4)/1920*log_moneyness**4))*    (zeta/x_zeta)*    (1+(((1-self.beta)**2)/24*self.sigma_0**2/(f_avg**(2-2*self.beta))     +0.25*self.rho*self.beta*self.vov*self.sigma_0/(f_avg**(1-self.beta))     +(2-3*self.rho**2)/24*self.vov**2)*self.T_expiry)

# ### Hagan normal implied vol expansion
class SABR_Hagan_N(SABR_base_model):
    """SABR model using normal implied vol quoting from Hagan's formula
    """
    def __init__(self, 
                 beta: float=1.0,
                 vov: float=1.0,
                 rho: float=0.0,
                 ATM=None,
                 sigma_0=None,
                 f=None,
                 T_expiry:float =1.0,
                 marking_mode: str='ATM',
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50,
                ):
        super().__init__(beta=beta, 
                         vov=vov,
                         rho=rho,
                         ATM=ATM,
                         sigma_0=sigma_0,
                         f=f,
                         T_expiry=T_expiry,
                         vol_type='N',
                         marking_mode=marking_mode,
                         logmoneyness_lo=logmoneyness_lo,
                         logmoneyness_hi=logmoneyness_hi,
                         K_lo=K_lo,
                         K_hi=K_hi,
                         strike_type=strike_type,
                         n_strikes=n_strikes
                        )
    
    def __str__(self):
        return r'T_expiry={}y, $\beta$={:.2f}, vov={:.2%}, $\rho$={:.2%}, ATM={:.2f}bps, f={:.2%}, $\sigma_0$={:.2%}'    .format(self.T_expiry, self.beta, self.vov, self.rho, self.ATM/ONE_BP, self.f, self.sigma_0)  
    
    @property
    def model_name(self):
        return 'SABR_Hagan_N'
    
    def smile_func(self, K):
        """Hagan normal smile approximation around ATM as written in the
        original 'Managing Smile Risk' paper (A.69a)
        K: strike
        """
        if np.abs(K-self.f)<ONE_BP/100:
            return self.ATM
        else:
            f_avg = np.sqrt(self.f*K)
            log_moneyness = np.log(self.f/K)
            zeta = self.vov/self.sigma_0*(f_avg)**(1-self.beta)*log_moneyness
            q_zeta = 1-2*self.rho*zeta+zeta**2
            x_zeta = np.log((np.sqrt(q_zeta)-self.rho+zeta)/(1-self.rho))
            g = -self.beta*(2-self.beta)/(24*(f_avg)**(2-2*self.beta))
            return self.sigma_0*(f_avg)**(self.beta)*        (1+1/24*log_moneyness**2+1/1920*log_moneyness**4)/        (1         +((1-self.beta)**2)/24*log_moneyness**2         +((1-self.beta)**4)/1920*log_moneyness**4)*    (zeta/x_zeta)*    (1+     (g*self.sigma_0**2      +0.25*self.rho*self.vov*self.sigma_0*self.beta/(f_avg**(1-1*self.beta))      +(2-3*self.rho**2)/24*self.vov**2)*self.T_expiry)
        
    def smile_func_full(self, K):
        """Hagan normal smile approximation around ATM as written in the
        original 'Managing Smile Risk' paper (A.67a)
        K: strike
        """
        if np.abs(K-self.f)<ONE_BP/100:
            return self.ATM
        else:
            #local_vol_int = (self.f**(1-self.beta)-K**(1-self.beta))/(1-self.beta)
            f_avg = np.sqrt(self.f*K)
            #f_avg = 0.5*(self.f+K)
            zeta = self.vov/self.sigma_0*(self.f-K)/(f_avg**self.beta)
            q_zeta = 1-2*self.rho*zeta+zeta**2
            x_zeta = np.log((np.sqrt(q_zeta)-self.rho+zeta)/(1-self.rho))
            g = -self.beta*(2-self.beta)/(24*(f_avg)**(2-2*self.beta))
            return self.sigma_0*(1-self.beta)*(self.f-K)        /(self.f**(1-self.beta)-K**(1-self.beta))*(zeta/x_zeta)*        (1+         (g*self.sigma_0**2+0.25*self.rho*self.vov*self.sigma_0*self.beta/(f_avg**(1-self.beta))          +(2-3*self.rho**2)/24*self.vov**2)         *self.T_expiry)

# Monte Carlo pricer, by discretizing the SABR SDEs
class SABR_MC(SABR_base_model):
    """SABR model with Monte Carlo pricing
    """
    def __init__(self,
                 beta: float=1.0,
                 vov: float=1.0,
                 rho: float=0.0,
                 ATM=None,
                 sigma_0=None,
                 f=None,
                 T_expiry: float=1.0,
                 vol_type: str='LN',
                 marking_mode: str='ATM',
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50,
                 n_sim: int = int(1e3),
                 scheme_steps: int=int(1e2)
                ) -> None:
        super().__init__(beta=beta, 
                         vov=vov,
                         rho=rho,
                         ATM=ATM,
                         sigma_0=sigma_0,
                         f=f,
                         T_expiry=T_expiry,
                         vol_type=vol_type,
                         marking_mode=marking_mode,
                         logmoneyness_lo=logmoneyness_lo,
                         logmoneyness_hi=logmoneyness_hi,
                         K_lo=K_lo,
                         K_hi=K_hi,
                         strike_type=strike_type,
                         n_strikes=n_strikes
                        )
        self._n_sim = n_sim
        self._scheme_steps = scheme_steps
    
    @property
    def model_name(self):
        return 'SABR_Hagan_MC'
    
    @property
    def n_sim(self):
        return self._n_sim
    @n_sim.setter
    def n_sim(self, new_n_sim):
        self._n_sim = new_n_sim
   
    @property
    def scheme_steps(self):
        return self._scheme_steps
    @scheme_steps.setter
    def scheme_steps(self, new_scheme_steps):
        self._scheme_steps = new_scheme_steps
        
    @property
    def MC_generator(self):
        return SABR(x0=[self.f, self.sigma_0], T=self.T_expiry,                        scheme_steps=self._scheme_steps, keys=['F', 'vol'],                        beta=self.beta, vov=self.vov, rho=self.rho,                        barrier=[0,None], barrier_condition='absorb')
    
    @property
    @lru_cache(maxsize=None)
    def MC_paths(self):
        gen = self.MC_generator
        F_terminal_samples = []
        for i in range(self.n_sim):
            df = gen.simulate()
            F_terminal_samples.append(df['F'].iloc[-1])
        return F_terminal_samples
        
    def clear_MC_cache(self):
        self.MC_paths.cache_clear()

    def payoff_helper(self, payoff):
        """Returns a lambda with the specification of the payoff name
        """
        if payoff == 'Call':
            return lambda X, K: max(X-K,0)
        elif payoff == 'Put':
            return lambda X, K: max(K-X, 0)
        else:
            raise NameError( "Unsupported payoff name : {}".format(payoff) )

    def option_price(self, K, payoff='Call', regen_paths=False):
        """Returns the call/put price estimated from Monte Carlo simulations
        of the spot/vol SABR dynamics
        """        
        payoff_func = self.payoff_helper(payoff)
        payoff_sims = []
        
        if regen_paths:
            gen = self.MC_generator
            for i in range(self.n_sim):
                df = gen.simulate()
                F_terminal = df['F'].iloc[-1]
                payoff_sims.append(payoff_func(F_terminal, K))
        else:
            F_terminal_samples = self.MC_paths
            for F_terminal in F_terminal_samples:
                payoff_sims.append(payoff_func(F_terminal, K))
        return np.mean(payoff_sims)
            
    def smile_func(self, K):
        """Implied vol smile converted from option Monte Carlo prices
        K: strike
        """
        if K <= self.f:
            payoff = 'Put'
        else:
            payoff = 'Call'
            
        price = self.option_price(K, payoff=payoff)
        return self.IV.vol_from_price(price, self.f, K, self.T_expiry, payoff=payoff)

# Normal SABR
# SABR formula for $\beta=0$ with an adjusted $\sigma_0$ to account for the actual local vol in the model.
class SABR_Goes_Normal(SABR_base_model):
    """SABR model with beta = 0 and ajusted sigma_0 to account for the actual local vol.
    Lognormal quoting.
    """
    def __init__(self, 
                 beta: float=1.0,
                 vov: float=1.0,
                 rho: float=0.0,
                 ATM=None,
                 sigma_0=None,
                 f=None,
                 T_expiry: float=1.0,
                 vol_type: str='LN',
                 marking_mode: str='ATM',
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50,
                 n_integral: int=1
                ) -> None:
        super().__init__(beta=beta, 
                         vov=vov,
                         rho=rho,
                         ATM=ATM,
                         sigma_0=sigma_0,
                         f=f,
                         T_expiry=T_expiry,
                         vol_type=vol_type,
                         marking_mode=marking_mode,
                         logmoneyness_lo=logmoneyness_lo,
                         logmoneyness_hi=logmoneyness_hi,
                         K_lo=K_lo,
                         K_hi=K_hi,
                         strike_type=strike_type,
                         n_strikes=n_strikes
                        )
        # number of discretization step of the integral in the approximation formula 
        self._n_integral = n_integral
        
    @property
    def n_integral(self):
        return self._n_integral
    @n_integral.setter
    def n_integral(self, new_n_integral):
        self._n_integral = new_n_integral
    
    @property
    def step_integral(self):
        return self.T_expiry/self._n_integral
        
    def sigma_0_effective(self, K):
        """Adjusted initial vol to match first order of ATM expansion when using
        normal SABR as base model
        """
        if np.abs(K-self.f)<ONE_BP/100:
            return self.sigma_0*(self.f**self.beta)
        else:
            return self.sigma_0*(1-self.beta)*(K-self.f)/(K**(1-self.beta)-self.f**(1-self.beta))
    
    def vov_scaled(self, K):
        """Helper function for the call/put price approximation
        """
        return self.vov/self.sigma_0_effective(K)
    
    def q(self, z, K):
        """Helper function for the call/put price approximation
        """
        return 1-2*self.rho*self.vov_scaled(K)*z+(self.vov_scaled(K)**2)*(z**2)

    def J(self, x, K):
        """Helper function for the call/put price approximation
        """
        X_0 = self.f-K
        num = np.sqrt(self.q(x, K))-self.rho+self.vov_scaled(K)*x
        denom = np.sqrt(self.q(X_0, K))-self.rho+self.vov_scaled(K)*X_0
        return 1/(self.vov)*np.log(num/denom)

    def dJ(self, x, K):
        """Helper function for the call/put price approximation
        """
        return 1/(self.sigma_0_effective(K)*np.sqrt(self.q(x, K)))
        
    def local_time(self, t, x, K=None):
        """Density of (S_t-K)/(alpha_t) at x under the stochastic vol measure
        """
        return 1/np.sqrt(2*np.pi*t)*self.dJ(x, K)*np.exp(-self.J(x, K)**2/(2*t))
    
    def integrand(self, K, t, x):
        """Integrand to compute E[J^-1(W_t)], used as an adjustment for the missing drift
        """
        return self.local_time(t, x, K)*x
    
    def option_price(self, K, payoff='Call'):
        """Returns the call/put price approximation derived for normal SABR
        in "SABR Goes Normal".
        """
        if payoff == "Call":
            return self.call_price(K)
        elif payoff == "Put":
            return self.call_price(K) + (K-self.f)
    
    def int_local_time(self, t, K):
        """Explicit form for the integral of the local time between 0 and t
        """
        dJ = self.dJ(0, K)
        J = self.J(0, K)
        return dJ*np.exp(-J**2/(2*t))*np.sqrt(2/np.pi)*np.sqrt(t)    +dJ*J*(np.erf(J/(np.sqrt(2*t)))-1)
    
    def call_price(self, K):
        """Returns the call price approximation derived from normal SABR
        """
        X_0 = self.f-K
        intrinsic_value = max(X_0, 0)

        #s = self.int_local_time(self.T_expiry, K)
        
        s = 0
        time_steps = [t*self.step_integral for t in range(self.n_integral)]
        for t in time_steps:
            t_next = t + self.step_integral
            t_mid = 0.5*(t+t_next)
        
            f = partial( self.integrand, K, t_mid )
            L = X_0 - quad(f, -np.inf, np.inf, limit=10)[0]
            s += self.local_time(t_mid, -L, K=K)*self.step_integral
        return intrinsic_value+0.5*self.sigma_0_effective(K)**2*s

class SABR_Goes_Normal_LN(SABR_Goes_Normal):
    """SABR model with beta = 0 and ajusted sigma_0 to account for the actual local vol.
    Lognormal quoting.
    """
    def __init__(self, 
                 beta: float=1.0,
                 vov: float=1.0,
                 rho: float=0.0,
                 ATM=None,
                 sigma_0=None,
                 f=None,
                 T_expiry: float=1.0,
                 marking_mode: str='ATM',
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50,
                 n_integral: int=1
                ) -> None:
        super().__init__(beta=beta,
                         vov=vov, 
                         rho=rho,
                         ATM=ATM,
                         sigma_0=sigma_0,
                         f=f,
                         T_expiry=T_expiry,
                         vol_type='LN',
                         marking_mode=marking_mode,
                         logmoneyness_lo=logmoneyness_lo,
                         logmoneyness_hi=logmoneyness_hi,
                         K_lo=K_lo,
                         K_hi=K_hi,
                         strike_type=strike_type,
                         n_strikes=n_strikes,
                         n_integral=n_integral
                        )
    
    @property
    def model_name(self):
        return 'SABR_Goes_Normal_LN'
    
    def smile_func(self, K):
        """Hagan normal smile approximation around ATM for beta = 0 as written in the
        original 'Managing Smile Risk' paper (A.70b)
        K: strike
        """
        b_0 = self.sigma_0_effective(K)
        if np.abs(K-self.f)<ONE_BP/100:
            return b_0/self.f*(1                               + ((b_0**2)/(24*self.f**2) + (2-3*self.rho**2)/24*self.vov**2)                               *self.T_expiry)
        else:
            log_moneyness = np.log(self.f/K)
            f_avg = np.sqrt(self.f*K)
            zeta = self.vov/b_0*f_avg*log_moneyness
            q_zeta = 1-2*self.rho*zeta+zeta**2
            x_zeta = np.log((np.sqrt(q_zeta)-self.rho+zeta)/(1-self.rho))

            return b_0*log_moneyness/(self.f-K)*zeta/x_zeta*                (1 + ((b_0**2)/(24*self.f*K) + (2-3*self.rho**2)/24*self.vov**2)*self.T_expiry)

class SABR_Goes_Normal_N(SABR_Goes_Normal):
    """SABR model with beta = 0 and ajusted sigma_0 to account for the actual local vol.
    Normal quoting.
    """
    def __init__(self, 
                 beta: float=1.0,
                 vov: float=1.0,
                 rho: float=0.0,
                 ATM=None,
                 sigma_0=None,
                 f=None,
                 T_expiry: float=1.0,
                 marking_mode: str='ATM',
                 logmoneyness_lo=None,
                 logmoneyness_hi=None,
                 K_lo=None,
                 K_hi=None,
                 strike_type: str='logmoneyness',
                 n_strikes: int=50,
                 n_integral: int=1
                ) -> None:
        super().__init__(beta=beta, 
                         vov=vov,
                         rho=rho,
                         ATM=ATM,
                         sigma_0=sigma_0,
                         f=f,
                         T_expiry=T_expiry,
                         vol_type='N',
                         marking_mode=marking_mode,
                         logmoneyness_lo=logmoneyness_lo,
                         logmoneyness_hi=logmoneyness_hi,
                         K_lo=K_lo,
                         K_hi=K_hi,
                         strike_type=strike_type,
                         n_strikes=n_strikes,
                         n_integral=n_integral,
                        )
    
    @property
    def model_name(self):
        return 'SABR_Goes_Normal_N'
    
    def smile_func(self, K):
        """Hagan normal smile approximation around ATM for beta = 0 as written in the
        original 'Managing Smile Risk' paper (A.70a)
        K: strike
        """
        b_0 = self.sigma_0_effective(K)
        if np.abs(K-self.f)<ONE_BP/100:
            return b_0*(1                        + ((2-3*self.rho**2)/24*self.vov**2)*self.T_expiry)
        else:
            log_moneyness = np.log(self.f/K)
            f_avg = np.sqrt(self.f*K)
            zeta = self.vov/b_0*f_avg*log_moneyness
            q_zeta = 1-2*self.rho*zeta+zeta**2
            x_zeta = np.log((np.sqrt(q_zeta)-self.rho+zeta)/(1-self.rho))

            return b_0*zeta/x_zeta*        (1+((2-3*self.rho**2)/24*self.vov**2)*self.T_expiry)