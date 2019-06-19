#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.misc import derivative
from scipy.optimize import brentq
from scipy.integrate import quad
from functools import partial
from functools import lru_cache
from ito_diffusion_multi_d import SABR
import abc


# In[2]:


ONE_BP = 1e-4
ONE_PCT = 1e-2


# ## Implied vol Black/normal quoter
# 
# Handles vol --> price and price --> vol conversions for European calls and puts
# 

# In[3]:


class Implied_vol():
    def __init__(self, vol_type='LN'):
        self._vol_type = vol_type
        if self._vol_type not in ['LN', 'N']:
            raise NameError( "Unsupported vol type : {}".format(self._vol_type) )
            
    def price_from_vol_LN(self, vol, f, K, T_expiry, payoff='Call'):
        """
        Black-Scholes Call/Put forward price from volatility, forward, strike, maturity
        """
        d1          = 1 / ( vol * np.sqrt( T_expiry ) ) * ( np.log( f / K ) + ( 0.5 * vol ** 2 ) * T_expiry )
        d2          = d1 - vol * np.sqrt( T_expiry ) 
        CallPrice   = (f * norm.cdf( d1 ) - K * norm.cdf( d2 ))

        if payoff == "Call":
           return CallPrice
        elif payoff == "Put":
           return CallPrice + (K-f)
        else:
            raise NameError( "Unsupported payoff : {}".format(payoff) )

    def price_from_vol_N(self, vol, f, K, T_expiry, payoff='Call'):
        """
        Call/Put forward price from normal implied volatility, forward, strike, maturity
        """
        total_var   = vol * np.sqrt(T_expiry)
        d           = (f-K)/total_var

        if payoff == "Call":
           return (f-K) * norm.cdf(d) + total_var*norm.pdf(d)
        elif payoff == "Put":
           return (K-f) * norm.cdf(-d) + total_var*norm.pdf(d)
        else:
            raise NameError( "Unsupported payoff : {}".format(payoff) )
            
    def price_from_vol(self, vol, f, K, T_expiry, payoff='Call'):
        """
        Black-Scholes Call/Put price from volatility, forward, strike, maturity
        """
        if self._vol_type == 'LN':
            return self.price_from_vol_LN(vol, f, K, T_expiry, payoff=payoff)
        elif self._vol_type == 'N':
            return self.price_from_vol_N(vol, f, K, T_expiry, payoff=payoff)
                
    def vol_from_price(self, price, f, K, T_expiry, payoff='Call'): 
        """
        Black-Scholes Call/Put implied volatility from price, the rest of the parameters (strike, rates, maturity) 
        are provided in the deal terms
        """
        def target_func( price, vol ):
            return self.price_from_vol(vol, f, K, T_expiry, payoff=payoff) - price 
                     
        try:
            implied_vol = brentq( partial( target_func, price ), 1e-6, 1.5 )
        except:
            print('Price: {}, strike: {}, payoff: {}'.format(price, K, payoff))


# ## Generic vol model
# 
# Handles option pricing and vol surface interpolation

# In[4]:


class Vol_model(abc.ABC):
    """Generic class for option quoting
    """
    def __init__(self, f=None, T_expiry=1, vol_type=None,                 moneyness_lo=-3, moneyness_hi=3, n_strikes=50):
        # initial forward
        self._f = f
        # maturity
        self._T_expiry = T_expiry
        
        # moneyness/strike boundaries
        self._moneyness_lo = moneyness_lo
        self._moneyness_hi = moneyness_hi
        
        # absolute boundaries in strike space
        self._K_abs_min = ONE_BP
        self._K_abs_max = 12 * ONE_PCT
        
        # number of knots in the smile construction
        self._n_strikes = n_strikes
        
        # normal or Lognormal implied vol
        self._vol_type = vol_type
        
    @property
    def f(self):
        return self._f
    @f.setter
    def f(self, new_f):
        self._f = new_f

    @property
    def T_expiry(self):
        return self._T_expiry
    @T_expiry.setter
    def T_expiry(self, new_T_expiry):
        self._T_expiry = new_T_expiry
        
    @property
    @abc.abstractmethod
    def label(self):
        pass
    
    @property
    @abc.abstractmethod
    def ATM_LN(self):
        pass
        
    @property
    def total_var(self):
        return self.ATM_LN*np.sqrt(self.T_expiry)
        
    @property
    def moneyness_lo(self):
        return self._moneyness_lo
    @moneyness_lo.setter
    def moneyness_lo(self, new_value):
        self._moneyness_lo = new_value
    
    @property
    def moneyness_hi(self):
        return self._moneyness_hi
    @moneyness_hi.setter
    def moneyness_hi(self, new_value):
        self._moneyness_hi = new_value
        
    @property
    def K_lo(self):
        return max(self.f * np.exp(self.moneyness_lo * self.total_var), self._K_abs_min)
    @property
    def K_hi(self):
        return min(self.f * np.exp(self.moneyness_hi * self.total_var), self._K_abs_max)
   
    @property
    def n_strikes(self):
        return self._n_strikes
    @n_strikes.setter
    def n_strikes(self, new_value):
        self._n_strikes = new_value
    
    @property
    def strike_grid(self):
        return np.linspace(self.K_lo, self.K_hi, self.n_strikes)
    @property
    def logmoneyness_grid(self):
        return list(map(lambda K: np.log(K/self.f)/self.total_var,                        self.strike_grid))
    
    @property
    def vol_type(self):
        return self._vol_type
        
    @property
    def IV(self):
        return Implied_vol(vol_type=self.vol_type)

    @abc.abstractmethod
    def smile_func(self, K):
        """To be defined by a model
        """
        pass
    
    @property
    #@lru_cache(maxsize=None)
    def smile(self):
        """Returns a dictionary strike -> implied vol
        """
        return dict(zip(self.strike_grid,                        list(map(lambda K: self.smile_func(K), self.strike_grid))))
    
    def option_price(self, K, payoff='Call'):
        """Returns the call/put price corresponding the implied vol at strike K 
        """
        return self.IV.price_from_vol(self.smile_func(K), self.f, K, self.T_expiry,                                      payoff=payoff)
    
    @property
    #@lru_cache(maxsize=None)
    def option_price_curve(self,  payoff='Call'):
        """Returns a dictionary strike -> option price
        """
        return dict(zip(self.strike_grid,                        list(map(lambda K: self.option_price(K, payoff=payoff),                        self.strike_grid))))
    
    def pdf(self, K):
        """Probability density function of the underlyer, derived by differentiating twice
        the model implied call prices with respect to the strike.
        """
        return derivative(partial(self.option_price, payoff='Call'),                               K,                               dx=ONE_BP/10,
                               n=2,
                               order=3)        
   
    @property
    #@lru_cache(maxsize=None)
    def pdf_curve(self):
        """Returns a dictionary strike -> probability density function 
        of the terminal distribution of the underlyer
        """
        return dict(zip(self.strike_grid,                        list(map(lambda K: self.pdf(K), self.strike_grid))))
        
    def plot(self, log_moneyness=False, curve_type='smile',            font_size=32, legend_size=32):
        if log_moneyness:
            x_grid = self.logmoneyness_grid
            xlabel = 'logmoneyness'
        else:
            x_grid = self.strike_grid
            xlabel = 'strike'
            
        if curve_type == 'smile':
            values = self.smile.values()
            ylabel = 'implied vol'
            label = self.label
        elif curve_type == 'pdf':
            values = self.pdf_curve.values()
            ylabel = 'pdf'
            label = self.label + r', expiry={}y'.format(self.T_expiry)
        else:
            raise NameError( "Unsupported curve type : {}".format(curve_type) )
        
        fig, ax = plt.subplots(figsize=(29, 21), nrows=1, ncols=1)
        ax.plot(x_grid, values, label=label)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.legend(loc='upper right', prop={'size': legend_size})
        ax.tick_params(labelsize=font_size)
        plt.tight_layout()
        plt.show()
        
    def smile_plot(self, log_moneyness=False,                  font_size=32, legend_size=32):
        self.plot(log_moneyness=log_moneyness, curve_type='smile',                 font_size=font_size, legend_size=legend_size)
    
    def pdf_plot(self, log_moneyness=False,                font_size=32, legend_size=32):
        return self.plot(log_moneyness=log_moneyness, curve_type='pdf',                        font_size=font_size, legend_size=legend_size)


# ## Displaced lognormal
# 
# Pure local volatility model here

# In[5]:


class SLN_adjustable_backbone(Vol_model):
    """Shifted lognormal model with adjustable backbone as developped 
    in Andersen and Piterbarg in (16.5):
    dF_t = sigma*(shift*F_t+(mixing-shift)*F_0+(1-m)*L)*dW_t
    """
    def __init__(self, sigma=1, shift=0, mixing=0, L=0,                 f=None, T_expiry=1, vol_type=None,                 moneyness_lo=-3, moneyness_hi=3, n_strikes=50):
        
        super().__init__(f=f, T_expiry=T_expiry, vol_type=vol_type,                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         n_strikes=n_strikes)
        self._sigma = sigma
        self._shift = shift
        self._mixing = mixing
        self._L = L
        
    @property
    def sigma(self):
        return self._sigma
    @sigma.setter
    def sigma(self, new_sigma):
        self._sigma = float(new_sigma)
    
    @property
    def shift(self):
        return self._shift
    @shift.setter
    def shift(self, new_shift):
        self._shift = new_shift
     
    @property
    def mixing(self):
        return self._mixing
    @mixing.setter
    def mixing(self, new_mixing):
        self._mixing = new_mixing
        
    @property
    def L(self):
        return self._L
    @L.setter
    def L(self, new_L):
        self._L = new_L
        
    @property
    def label(self):
        return r'$\sigma$={:.2}, shift={:.0%}, $\mixing$={:.0%}, L={:.2}, f={:.2%}'        .format(self.sigma, self.shift, self.mixing, self.L, self.f)
    
    @abc.abstractmethod
    def smile_func(self, K):
        pass


# 

# In[6]:


class SLN_adjustable_backbone_LN(SLN_adjustable_backbone):
    """Shifted lognormal model with adjustable backbone in lognormal quoting
    """
    def __init__(self, sigma=1, shift=0, mixing=0, L=0,                 f=None, T_expiry=1,                 moneyness_lo=-3, moneyness_hi=3, n_strikes=50):
        super().__init__(sigma=sigma, shift=shift, mixing=mixing, L=L,                         f=f, T_expiry=T_expiry, vol_type='LN',                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         n_strikes=n_strikes)
        
    @property
    def ATM_LN(self):
        local_vol_f = self.mixing*self.f+(1-self.mixing)*self.L
        return self.sigma*(self.mixing*self.f+(1-self.mixing)*self.L)/self.f    *(1+1/24*((-(self.shift/local_vol_f)**2+1/(self.f**2))*(self.sigma**2)*(local_vol_f**2))      *self.T_expiry)
        
    @property
    def ATM(self):
        return self.ATM_LN
    
    def smile_func(self, K):
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


# ## SABR generic class
# Possible SABR calculations include : 
# * Hagan normal implied vol
# * Hagan lognormal implied
# * Monte Carlo

# In[7]:


class SABR_base_model(Vol_model):
    """Generic class for SABR model
    """
    def __init__(self, beta=1, vov=1, rho=0,                 ATM=None, sigma_0=None, f=None, T_expiry=1,                 vol_type=None, marking_mode='ATM',                 moneyness_lo=-3, moneyness_hi=3, n_strikes=50):
        """Implementation of SABR implied vol and approximated price formula.
        beta, vov and rho are marked.
        One can either mark the ATM vol and the forward and solve for sigma_0,
        or mark sigma_0 directly.
        """
        super().__init__(f=f, T_expiry=T_expiry, vol_type=vol_type,                              moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                              n_strikes=n_strikes)
        self._beta = float(beta)
        self._vov = float(vov)
        self._rho = float(rho)
        
        # ATM or sigma_0
        self._marking_mode = marking_mode
    
        self._ATM = ATM
        self._sigma_0 = sigma_0

        if self._marking_mode == 'ATM' and self._ATM == None:
            raise NameError( "ATM must be marked in ATM marking mode" )
        elif self._marking_mode == 'sigma_0' and self._sigma_0 == None:
            raise NameError( "sigma_0 must be marked in sigma_0 marking mode" )

    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self, new_beta):
        self._beta = float(new_beta)
    
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
    def label(self):
        """Used to decorate plot
        """
        return r'$\beta$={:.2}, vov={:.0%}, $\rho$={:.0%}, ATM={:.2%}, f={:.2%}, $\sigma_0$={:.2%}'        .format(self.beta, self.vov, self.rho, self.ATM, self.f, self.sigma_0)
        
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

# In[8]:


class SABR_Hagan_LN(SABR_base_model):
    """SABR model using lognormal implied vol quoting from Hagan's formula
    """
    def __init__(self, beta=1, vov=1, rho=0,                 ATM=None, sigma_0=None, f=None, T_expiry=1,                 marking_mode='ATM',                 moneyness_lo=-3, moneyness_hi=3, n_strikes=50):
        super().__init__(beta=beta, vov=vov, rho=rho,                         ATM=ATM, sigma_0=sigma_0, f=f, T_expiry=T_expiry,                         vol_type='LN', marking_mode=marking_mode,                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         n_strikes=n_strikes)
        
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

# In[9]:


class SABR_Hagan_N(SABR_base_model):
    """SABR model using normal implied vol quoting from Hagan's formula
    """
    def __init__(self, beta=1, vov=1, rho=0,                 ATM=None, sigma_0=None, f=None, T_expiry=1,                 marking_mode='ATM',                 moneyness_lo=-3, moneyness_hi=3, n_strikes=50):
        super().__init__(beta=beta, vov=vov, rho=rho,                         ATM=ATM, sigma_0=sigma_0, f=f, T_expiry=T_expiry,                         vol_type='N', marking_mode=marking_mode,                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         n_strikes=n_strikes)
        
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


# ### Monte Carlo pricer, by discretizing the SABR SDEs

# In[10]:


class SABR_MC(SABR_base_model):
    """SABR model with Monte Carlo pricing
    """
    def __init__(self, beta=1, vov=1, rho=0,                 ATM=None, sigma_0=None, f=None, T_expiry=1,                 vol_type='LN', marking_mode='ATM',                 moneyness_lo=-3, moneyness_hi=3, n_strikes=50,                 n_sim = int(1e3), scheme_steps=int(1e2)):
        super().__init__(beta=beta, vov=vov, rho=rho,                         ATM=ATM, sigma_0=sigma_0, f=f, T_expiry=T_expiry,                         vol_type=vol_type, marking_mode=marking_mode,                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         n_strikes=n_strikes)
        self._n_sim = n_sim
        self._scheme_steps = scheme_steps
        
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
        if K <= self.ATM:
            payoff = 'Put'
        else:
            payoff = 'Call'
            
        price = self.option_price(K, payoff=payoff)
        return self.IV.vol_from_price(price, self.f, K, self.T_expiry, payoff=payoff)


# ## Normal SABR
# 
# SABR formula for $\beta=0$ with an adjusted $\sigma_0$ to account for the actual local vol in the model.

# In[11]:


class SABR_Goes_Normal(SABR_base_model):
    """SABR model with beta = 0 and ajusted sigma_0 to account for the actual local vol.
    Lognormal quoting.
    """
    def __init__(self, beta=0, vov=1, rho=0,                 ATM=None, sigma_0=None, f=None, T_expiry=1,                 vol_type='LN', marking_mode='ATM',                 moneyness_lo=-3, moneyness_hi=3, n_strikes=50,                 n_integral=1, moneyness_switch_lo=-2, moneyness_switch_hi=2):
        super().__init__(beta=beta, vov=vov, rho=rho,                         ATM=ATM, sigma_0=sigma_0, f=f, T_expiry=T_expiry,                         vol_type=vol_type, marking_mode=marking_mode,                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         n_strikes=n_strikes)
        # number of discretization step of the integral in the approximation formula 
        self._n_integral = n_integral
        
        # boundaries to determine the choice of pricing model
        # around ATM : price from adjusted normal vol expansion
        # outside ATM : local vol integration
        self._moneyness_switch_lo = moneyness_lo
        self._moneyness_switch_hi = moneyness_switch_hi
    
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
        return 1/(self.vov_scaled(K)*self.sigma_0_effective(K))*np.log(num/denom)

    def dJ(self, x, K):
        """Helper function for the call/put price approximation
        """
        num = (self.vov_scaled(K)**2*x-self.vov_scaled(K)*self.rho)/np.sqrt(self.q(x, K))        +self.vov_scaled(K)
        denom = np.sqrt(self.q(x, K))-self.rho+self.vov_scaled(K)*x
        return 1/(self.vov_scaled(K)*self.sigma_0_effective(K))*num/denom
    
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
        if np.log(self.f/K) > self._moneyness_switch_lo        and np.log(self.f/K) < self._moneyness_switch_hi:
            return self.IV.price_from_vol(self.smile_func(K), self.f, K, self.T_expiry,                                          payoff=payoff)
        else:
            if payoff == "Call":
                return self.call_price(K)
            elif payoff == "Put":
                return self.call_pric(K) + (K-self.f)
            
    def call_price(self, K):
        """Returns the call price approximation derived from normal SABR
        """
        X_0 = self.f-K
        intrinsic_value = max(X_0, 0)

        s = 0
        time_steps = [t*self.step_integral for t in range(self.n_integral)]

        for t in time_steps:
            t_next = t + self.step_integral
            t_mid = 0.5*(t+t_next)

            f = partial( self.integrand, K, t_mid )
            L = X_0 - quad(f, -np.inf, np.inf, limit=10)[0]

            s += self.local_time(t_mid, -L, K=K)*self.step_integral

        return intrinsic_value+0.5*self.sigma_0_effective(K)**2*s


# In[12]:


class SABR_Goes_Normal_LN(SABR_Goes_Normal):
    """SABR model with beta = 0 and ajusted sigma_0 to account for the actual local vol.
    Lognormal quoting.
    """
    def __init__(self, beta=0, vov=1, rho=0,                 ATM=None, sigma_0=None, f=None, T_expiry=1,                 marking_mode='ATM',                 moneyness_lo=-3, moneyness_hi=3, n_strikes=50,                 n_integral=1, moneyness_switch_lo=-1, moneyness_switch_hi=1):
        super().__init__(beta=beta, vov=vov, rho=rho,                         ATM=ATM, sigma_0=sigma_0, f=f, T_expiry=T_expiry,                         vol_type='LN', marking_mode=marking_mode,                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         n_strikes=n_strikes, n_integral=n_integral,                         moneyness_switch_lo=moneyness_switch_lo,                         moneyness_switch_hi=moneyness_switch_hi)
        
    def smile_func(self, K):
        """Hagan normal smile approximation around ATM for beta = 0 as written in the
        original 'Managing Smile Risk' paper (A.70a)
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


# In[13]:


class SABR_Goes_Normal_N(SABR_Goes_Normal):
    """SABR model with beta = 0 and ajusted sigma_0 to account for the actual local vol.
    Normal quoting.
    """
    def __init__(self, beta=0, vov=1, rho=0,                 ATM=None, sigma_0=None, f=None, T_expiry=1,                 marking_mode='ATM',                 moneyness_lo=-3, moneyness_hi=3, n_strikes=50,                 n_integral=1, moneyness_switch_lo=-1, moneyness_switch_hi=1):
        super().__init__(beta=beta, vov=vov, rho=rho,                         ATM=ATM, sigma_0=sigma_0, f=f, T_expiry=T_expiry,                         vol_type='N', marking_mode=marking_mode,                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         n_strikes=n_strikes, n_integral=n_integral,                         moneyness_switch_lo=moneyness_switch_lo,                         moneyness_switch_hi=moneyness_switch_hi)
        
    def smile_func(self, K):
        """Hagan normal smile approximation around ATM for beta = 0 as written in the
        original 'Managing Smile Risk' paper (A.70a)
        K: strike
        """
        b_0 = self.sigma_0_effective(K)
        log_moneyness = np.log(self.f/K)
        f_avg = np.sqrt(self.f*K)
        zeta = self.vov/b_0*f_avg*log_moneyness
        q_zeta = 1-2*self.rho*zeta+zeta**2
        x_zeta = np.log((np.sqrt(q_zeta)-self.rho+zeta)/(1-self.rho))
            
        return b_0*zeta/x_zeta*    (1+((2-3*self.rho**2)/24*self.vov**2)*self.T_expiry)


# ## Modified SABR with tanh local vol
# 
# In regular SABR, beta controls the backbone shape. Standard market practices are to have beta close to 1 in low rates environment (lognormal SABR) and closer to 0 when rates hike (normal SABR). 
# 
# tanh(x) has a unit slope close to zero and flattens for larger x, so using it as a local vol function allows to dynamically mirror the remarking of beta as rates move. Moreover, it yields closed-form vol expansion in Hagan's formula

# In[14]:


class SABR_tanh_base_model(Vol_model):
    """Generic class for SABR_tanh model
    dX_t = s_t*C(X_t)*dW_t
    ds_t = vov*s_t*dB_t
    d<W,B>_t = rho*dt
    C(x) = tanh(x/l)
    """
    def __init__(self, l=1, vov=1, rho=0,                 sigma_0=None, f=None, T_expiry=1,                 vol_type=None,                 moneyness_lo=-3, moneyness_hi=3, n_strikes=50):
        """Implementation of SABR implied vol and approximated price formula.
        l, vov and rho are marked.
        """
        super().__init__(f=f, T_expiry=T_expiry, vol_type=vol_type,                              moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                              n_strikes=n_strikes)
        self._l = float(l)
        self._vov = float(vov)
        self._rho = float(rho)
        self._sigma_0 = float(sigma_0)

    @property
    def l(self):
        return self._l
    @l.setter
    def l(self, new_l):
        self._l = float(new_l)
    
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
    
    # need to have a ATM_LN to calculate total_var, which is used for
    # the boundaries
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
    def label(self):
        """Used to decorate plot
        """
        return r'l={:.2}, vov={:.0%}, $\rho$={:.0%}, ATM={:.2%}, f={:.2%}, $\sigma_0$={:.2%}'        .format(self.l, self.vov, self.rho, self.ATM, self.f, self.sigma_0)
    
    def local_vol_inv_int(self, x):
        """Closed form for the primitive of 1/(tanh((x+s)/l))
        """
        return self.l*np.log(np.sinh(x/self.l))
    
    @abc.abstractmethod
    def smile_func(self, K):
        pass


# In[15]:


class SABR_tanh_LN(SABR_tanh_base_model):
    """SABR_tanh model using lognormal implied vol quoting
    """
    def __init__(self, l=1, vov=1, rho=0,                 sigma_0=None, f=None, T_expiry=1,                 moneyness_lo=-3, moneyness_hi=3, n_strikes=50):
        super().__init__(l=l, vov=vov, rho=rho,                         sigma_0=sigma_0, f=f, T_expiry=T_expiry,                         vol_type='LN',                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         n_strikes=n_strikes)
    
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


# In[16]:


class SABR_tanh_N(SABR_tanh_base_model):
    """SABR_tanh model using lognormal implied vol quoting
    """
    def __init__(self, l=1, vov=1, rho=0,                 sigma_0=None, f=None, T_expiry=1,                 moneyness_lo=-3, moneyness_hi=3, n_strikes=50):
        super().__init__(l=l, vov=vov, rho=rho,                         sigma_0=sigma_0, f=f, T_expiry=T_expiry,                         vol_type='N',                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         n_strikes=n_strikes)
    
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

