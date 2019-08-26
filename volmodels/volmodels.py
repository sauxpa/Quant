#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.misc import derivative
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.special import erfi
from scipy.special import erf
from functools import partial
from functools import lru_cache
from ito_diffusions import SABR
import abc

ONE_BP = 1e-4
ONE_PCT = 1e-2

# Implied vol Black/normal quoter
# 
# Handles vol --> price and price --> vol conversions for European calls and puts
# 
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
            return brentq( partial( target_func, price ), 1e-6, 1.5 )
        except:
            print('Price: {}, strike: {}, payoff: {}'.format(price, K, payoff))
        
# ## Generic vol model
# 
# Handles option pricing and vol surface interpolation
class Vol_model(abc.ABC):
    """Generic class for option quoting
    """
    def __init__(self, f=None, T_expiry=1, vol_type=None,                 moneyness_lo=None, moneyness_hi=None, K_lo=None, K_hi=None,                 strike_type='logmoneyness', n_strikes=50):
        # initial forward
        self._f = f
        # maturity
        self._T_expiry = T_expiry
        
        # strike of logmoneyness
        self._strike_type = strike_type
        
        # logmoneyness/strike boundaries, depending on strike_type
        self._moneyness_lo = moneyness_lo
        self._moneyness_hi = moneyness_hi
        self._K_lo = K_lo
        self._K_hi = K_hi
        
        # absolute boundaries in strike space
        self._K_abs_min = ONE_BP
        self._K_abs_max = 12 * ONE_PCT
        
        # number of knots in the smile construction
        self._n_strikes = n_strikes
        
        # normal or Lognormal implied vol
        self._vol_type = vol_type
        
    def __str__(self):
        return 'Generic vol model'
    
    @property
    @abc.abstractmethod
    def model_name(self):
        pass
    
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
    def moneyness_lo(self):
        if self.strike_type == 'strike':
            return self.K_lo/self.f
        else:
            return self._moneyness_lo
    @moneyness_lo.setter
    def moneyness_lo(self, new_moneyness_lo):
        if self.strike_type == 'strike':
            raise NameError('Impossible to mark moneyness in strike mode')
        else:
            self._moneyness_lo = new_moneyness_lo
    
    @property
    def moneyness_hi(self):
        if self.strike_type == 'strike':
            return self.K_hi/self.f
        else:
            return self._moneyness_hi
    @moneyness_hi.setter
    def moneyness_hi(self, new_moneyness_hi):
        if self.strike_type == 'strike':
            raise NameError('Impossible to mark moneyness in strike mode')
        else:
            self._moneyness_hi = new_moneyness_hi
        
    @property
    def K_lo(self):
        if self.strike_type == 'logmoneyness':
            return max(self.f * self.moneyness_lo, self._K_abs_min)
        else:
            return self._K_lo
    @K_lo.setter
    def K_lo(self, new_K_lo):
        if self.strike_type == 'logmoneyness':
            raise NameError('Impossible to mark strikes in logmoneyness mode')
        else:
            self._K_lo = new_K_lo
        
    @property
    def K_hi(self):
        if self.strike_type == 'logmoneyness':
            return min(self.f * self.moneyness_hi, self._K_abs_max)
        else:
            return self._K_hi
    @K_hi.setter
    def K_hi(self, new_K_hi):
        if self.strike_type == 'logmoneyness':
            raise NameError('Impossible to mark strikes in logmoneyness mode')
        else:
            self._K_hi = new_K_hi
            
    @property
    def n_strikes(self):
        return self._n_strikes
    @n_strikes.setter
    def n_strikes(self, new_value):
        self._n_strikes = new_value
    
    @property
    def strike_type(self):
        return self._strike_type
    @strike_type.setter
    def strike_type(self, new_strike_type):
        if new_strike_type in ['srike', 'logmoneyness']:
            self._strike_type = new_strike_type
        else:
            raise NameError("Unsupported strike type : {}".format(new_strike_type))

    @property
    def strike_grid(self):
        if self.strike_type == 'strike':
            return np.linspace(self.K_lo, self.K_hi, self.n_strikes)
        else:
            return list(map(lambda m: self.f*np.exp(m), self.logmoneyness_grid))
    @property
    def logmoneyness_grid(self):
        if self.strike_type == 'strike':
            return list(map(lambda K: np.log(K/self.f), self.strike_grid))
        else:
            return np.linspace(self.moneyness_lo, self.moneyness_hi, self.n_strikes)
    
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
        the model implied call pric+++++++++++++++++++++++++++++++es with respect to the strike.
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
        elif curve_type == 'pdf':
            values = self.pdf_curve.values()
            ylabel = 'pdf'
        else:
            raise NameError( "Unsupported curve type : {}".format(curve_type) )
        
        fig, ax = plt.subplots(figsize=(29, 21), nrows=1, ncols=1)
        ax.plot(x_grid, values, label=self.__str___())
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.legend(loc='upper right', prop={'size': legend_size})
        ax.tick_params(labelsize=font_size)
        plt.tight_layout()
        plt.show()
        
    def smile_plot(self, log_moneyness=False):
        self.plot(log_moneyness=log_moneyness, curve_type='smile')
    
    def pdf_plot(self, log_moneyness=False):
        return self.plot(log_moneyness=log_moneyness, curve_type='pdf')

# ## SVI 
# 
# Stochastic Volatility Inspired
# Not a diffusion model but rather a vol surface parametrization
class SVI_Raw(Vol_model):
    """Gatheral's raw SVI
    w(k) = a + b * ( rho * ( k - m ) + sqrt( ( k - m ) ^ 2 + sigma ^ 2 ) )
    where w(k) = sigma^2(k)*T_expiry is the total variance
    """
    def __init__(self, 
                 a=1, 
                 b=0,
                 rho=0, 
                 m=0,
                 sigma=0,
                 f=None, 
                 T_expiry=1, 
                 vol_type=None,
                 moneyness_lo=None, 
                 moneyness_hi=None, 
                 K_lo=None, 
                 K_hi=None,
                 strike_type='logmoneyness', 
                 n_strikes=50
                ):
        super().__init__(
            f=f, 
            T_expiry=T_expiry, 
            vol_type=vol_type,                         
            moneyness_lo=moneyness_lo, 
            moneyness_hi=moneyness_hi,
            K_lo=K_lo, 
            K_hi=K_hi,
            strike_type=strike_type, 
            n_strikes=n_strikes
        )
        self._a = a
        self._b = b
        self._rho = rho
        self._m = m
        self._sigma = sigma
    
    @property
    def sigma(self):
        return self._sigma
    @sigma.setter
    def sigma(self, new_sigma):
        self._sigma = new_sigma
    
    @property
    def a(self):
        return self._a
    @a.setter
    def a(self, new_a):
        self._a = new_a
    
    @property
    def b(self):
        return self._b
    @b.setter
    def b(self, new_b):
        self._b = new_b

    @property
    def rho(self):
        return self._rho
    @rho.setter
    def rho(self, new_rho):
        self._rho = new_rho
        
    @property
    def m(self):
        return self._m
    @m.setter
    def m(self, new_m):
        self._m = new_m

    def __str__(self):
        return r'$a$={:.2f}, $b$={:.2f}, $\rho$={:.0%}, m={:.2f}, $\sigma$={:.2f}, f={:.2%}'        .format(self.a, self.b, self.rho, self.m, self.sigma, self.f)
    
    @abc.abstractmethod
    def smile_func(self, K):
        pass
    
class SVI_Raw_LN(SVI_Raw):
    """Gatheral's raw SVI in lognormal quoting
    """
    def __init__(self, 
                 a=1, 
                 b=0,
                 rho=0, 
                 m=0,
                 sigma=0,
                 f=None, 
                 T_expiry=1, 
                 moneyness_lo=None, 
                 moneyness_hi=None, 
                 K_lo=None, 
                 K_hi=None,
                 strike_type='logmoneyness', 
                 n_strikes=50
                ):
        super().__init__(
            a=a,
            b=b,
            rho=rho,
            m=m,
            sigma=sigma,
            f=f, 
            T_expiry=T_expiry, 
            vol_type='LN',                         
            moneyness_lo=moneyness_lo, 
            moneyness_hi=moneyness_hi,
            K_lo=K_lo, 
            K_hi=K_hi,
            strike_type=strike_type, 
            n_strikes=n_strikes
        )
        
    @property
    def model_name(self):
        return 'SVI_Raw_LN'
    
    @property
    def ATM_LN(self):
        total_var = self.a + self.b * (self.rho * (-self.m) + np.sqrt((-self.m) ** 2 + self.sigma ** 2))
        return np.sqrt(total_var/self.T_expiry)
        
    @property
    def ATM(self):
        return self.ATM_LN
    
    def smile_func(self, K):
        """Implied vol comes from the SVI parametrization of the total variance
        """
        k = np.log(K/self.f)
        total_var = self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma ** 2))
        return np.sqrt(total_var/self.T_expiry)
    
# ## Displaced lognormal
# 
# Pure local volatility model here
class SLN_adjustable_backbone(Vol_model):
    """Shifted lognormal model with adjustable backbone as developped 
    in Andersen and Piterbarg in (16.5):
    dF_t = sigma*(shift*F_t+(mixing-shift)*F_0+(1-m)*L)*dW_t
    """
    def __init__(self, sigma=1, shift=0, mixing=0, L=0,                 f=None, T_expiry=1, vol_type=None,                 moneyness_lo=None, moneyness_hi=None, K_lo=None, K_hi=None,
                 strike_type='logmoneyness', n_strikes=50):
        
        super().__init__(f=f, T_expiry=T_expiry, vol_type=vol_type,                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         K_lo=K_lo, K_hi=K_hi,                         strike_type=strike_type, n_strikes=n_strikes)
        self._sigma = sigma
        self._shift = shift
        self._mixing = mixing
        self._L = L
    
    @property
    def sigma(self):
        return self._sigma
    @sigma.setter
    def sigma(self, new_sigma):
        self._sigma = new_sigma
    
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
        
    def __str__(self):
        return r'$\sigma$={:.2f}, shift={:.0%}, $\mixing$={:.0%}, L={:.2f}, f={:.2%}'        .format(self.sigma, self.shift, self.mixing, self.L, self.f)
    
    @abc.abstractmethod
    def smile_func(self, K):
        pass

class SLN_adjustable_backbone_LN(SLN_adjustable_backbone):
    """Shifted lognormal model with adjustable backbone in lognormal quoting
    """
    def __init__(self, sigma=1, shift=0, mixing=0, L=0,                 f=None, T_expiry=1,                 moneyness_lo=None, moneyness_hi=None, K_lo=None, K_hi=None,                 strike_type='logmoneyness', n_strikes=50):
        super().__init__(sigma=sigma, shift=shift, mixing=mixing, L=L,                         f=f, T_expiry=T_expiry, vol_type='LN',                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         K_lo=K_lo, K_hi=K_hi,                         strike_type=strike_type, n_strikes=n_strikes)
        
    @property
    def model_name(self):
        return 'SLN_adjustable_backbone_LN'
    
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
class SABR_base_model(Vol_model):
    """Generic class for SABR model
    """
    def __init__(self, beta=1, vov=1, rho=0,                 ATM=None, sigma_0=None, f=None, T_expiry=1,                 vol_type=None, marking_mode='ATM',                 moneyness_lo=None, moneyness_hi=None, K_lo=None, K_hi=None,                 strike_type='logmoneyness', n_strikes=50):
        """Implementation of SABR implied vol and approximated price formula.
        beta, vov and rho are marked.
        One can either mark the ATM vol and the forward and solve for sigma_0,
        or mark sigma_0 directly.
        """
        super().__init__(f=f, T_expiry=T_expiry, vol_type=vol_type,                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         K_lo=K_lo, K_hi=K_hi,                         strike_type=strike_type, n_strikes=n_strikes)
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
    def __init__(self, beta=1, vov=1, rho=0,                 ATM=None, sigma_0=None, f=None, T_expiry=1,                 marking_mode='ATM',                 moneyness_lo=None, moneyness_hi=None, K_lo=None, K_hi=None,                 strike_type='logmoneyness', n_strikes=50):
        super().__init__(beta=beta, vov=vov, rho=rho,                         ATM=ATM, sigma_0=sigma_0, f=f, T_expiry=T_expiry,                         vol_type='LN', marking_mode=marking_mode,                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         K_lo=K_lo, K_hi=K_hi,                         strike_type=strike_type, n_strikes=n_strikes)
    
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
    def __init__(self, beta=1, vov=1, rho=0,                 ATM=None, sigma_0=None, f=None, T_expiry=1,                 marking_mode='ATM',                 moneyness_lo=None, moneyness_hi=None, K_lo=None, K_hi=None,                 strike_type='logmoneyness', n_strikes=50):
        super().__init__(beta=beta, vov=vov, rho=rho,                         ATM=ATM, sigma_0=sigma_0, f=f, T_expiry=T_expiry,                         vol_type='N', marking_mode=marking_mode,                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,
                         K_lo=K_lo, K_hi=K_hi,\
                         strike_type=strike_type, n_strikes=n_strikes)
    
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

# ### Monte Carlo pricer, by discretizing the SABR SDEs
class SABR_MC(SABR_base_model):
    """SABR model with Monte Carlo pricing
    """
    def __init__(self, beta=1, vov=1, rho=0,                 ATM=None, sigma_0=None, f=None, T_expiry=1,                 vol_type='LN', marking_mode='ATM',                 moneyness_lo=None, moneyness_hi=None, K_lo=None, K_hi=None,                 strike_type='logmoneyness', n_strikes=50,                 n_sim = int(1e3), scheme_steps=int(1e2)):
        super().__init__(beta=beta, vov=vov, rho=rho,                         ATM=ATM, sigma_0=sigma_0, f=f, T_expiry=T_expiry,                         vol_type=vol_type, marking_mode=marking_mode,                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         K_lo=K_lo, K_hi=K_hi,                         strike_type=strike_type, n_strikes=n_strikes)
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

# ## Normal SABR
# 
# SABR formula for $\beta=0$ with an adjusted $\sigma_0$ to account for the actual local vol in the model.
class SABR_Goes_Normal(SABR_base_model):
    """SABR model with beta = 0 and ajusted sigma_0 to account for the actual local vol.
    Lognormal quoting.
    """
    def __init__(self, beta=0, vov=1, rho=0,                 ATM=None, sigma_0=None, f=None, T_expiry=1,                 vol_type='LN', marking_mode='ATM',                 moneyness_lo=None, moneyness_hi=None, K_lo=None, K_hi=None,                 strike_type='logmoneyness', n_strikes=50,                 n_integral=1):
        super().__init__(beta=beta, vov=vov, rho=rho,                         ATM=ATM, sigma_0=sigma_0, f=f, T_expiry=T_expiry,                         vol_type=vol_type, marking_mode=marking_mode,                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         K_lo=K_lo, K_hi=K_hi,                         strike_type=strike_type, n_strikes=n_strikes)
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
    def __init__(self, beta=0, vov=1, rho=0,                 ATM=None, sigma_0=None, f=None, T_expiry=1,                 marking_mode='ATM',                 moneyness_lo=None, moneyness_hi=None, K_lo=None, K_hi=None,                 strike_type='logmoneyness', n_strikes=50,                 n_integral=1):
        super().__init__(beta=beta, vov=vov, rho=rho,                         ATM=ATM, sigma_0=sigma_0, f=f, T_expiry=T_expiry,                         vol_type='LN', marking_mode=marking_mode,                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         K_lo=K_lo, K_hi=K_hi,                         strike_type=strike_type, n_strikes=n_strikes, n_integral=n_integral)
    
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
    def __init__(self, beta=0, vov=1, rho=0,                 ATM=None, sigma_0=None, f=None, T_expiry=1,                 marking_mode='ATM',                 moneyness_lo=None, moneyness_hi=None, K_lo=None, K_hi=None,                 strike_type='logmoneyness', n_strikes=50,                 n_integral=1):
        super().__init__(beta=beta, vov=vov, rho=rho,                         ATM=ATM, sigma_0=sigma_0, f=f, T_expiry=T_expiry,                         vol_type='N', marking_mode=marking_mode,                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         K_lo=K_lo, K_hi=K_hi,                         strike_type=strike_type, n_strikes=n_strikes, n_integral=n_integral)
    
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


# ## Modified SABR with tanh local vol
# 
# In regular SABR, beta controls the backbone shape. Standard market practices are to have beta close to 1 in low rates environment (lognormal SABR) and closer to 0 when rates hike (normal SABR). 
# 
# tanh(x) has a unit slope close to zero and flattens for larger x, so using it as a local vol function allows to dynamically mirror the remarking of beta as rates move. Moreover, it yields closed-form vol expansion in Hagan's formula
class SABR_tanh_base_model(Vol_model):
    """Generic class for SABR_tanh model
    dX_t = s_t*C(X_t)*dW_t
    ds_t = vov*s_t*dB_t
    d<W,B>_t = rho*dt
    C(x) = tanh(x/l)
    """
    def __init__(self, l=1, vov=1, rho=0,                 sigma_0=None, f=None, T_expiry=1,                 vol_type=None,                 moneyness_lo=None, moneyness_hi=None, K_lo=None, K_hi=None,                 strike_type='logmoneyness', n_strikes=50):
        """Implementation of SABR implied vol and approximated price formula.
        l, vov and rho are marked.
        """
        super().__init__(f=f, T_expiry=T_expiry, vol_type=vol_type,                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         K_lo=K_lo, K_hi=K_hi,                         strike_type=strike_type, n_strikes=n_strikes)
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
    def __init__(self, l=1, vov=1, rho=0,                 sigma_0=None, f=None, T_expiry=1,                 moneyness_lo=None, moneyness_hi=None, K_lo=None, K_hi=None,                 strike_type='logmoneyness', n_strikes=50):
        super().__init__(l=l, vov=vov, rho=rho,                         sigma_0=sigma_0, f=f, T_expiry=T_expiry,                         vol_type='LN',                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         K_lo=K_lo, K_hi=K_hi,                         strike_type=strike_type, n_strikes=n_strikes)
    
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
    def __init__(self, l=1, vov=1, rho=0,                 sigma_0=None, f=None, T_expiry=1,                 moneyness_lo=None, moneyness_hi=None, K_lo=None, K_hi=None,                 strike_type='logmoneyness', n_strikes=50):
        super().__init__(l=l, vov=vov, rho=rho,                         sigma_0=sigma_0, f=f, T_expiry=T_expiry,                         vol_type='N',                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         K_lo=K_lo, K_hi=K_hi,                         strike_type=strike_type, n_strikes=n_strikes)
    
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

class SABR_AS_base_model(Vol_model):
    """Generic class for SABR_tanh model
    dX_t = s_t*C(X_t)*dW_t
    ds_t = vov*s_t*dB_t
    d<W,B>_t = rho*dt
    C(x) = exp(-c*log(y/K_max)^2)
    """
    def __init__(self, K_max=1, c=1, vov=1, rho=0,                 sigma_0=None, f=None, T_expiry=1,                 vol_type=None,                 moneyness_lo=None, moneyness_hi=None, K_lo=None, K_hi=None,                 strike_type='logmoneyness', n_strikes=50):
        """Implementation of SABR implied vol and approximated price formula.
        K_max, c, vov and rho are marked.
        """
        super().__init__(f=f, T_expiry=T_expiry, vol_type=vol_type,                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         K_lo=K_lo, K_hi=K_hi,                         strike_type=strike_type, n_strikes=n_strikes)
        self._c = c
        self._K_max = K_max
        self._vov = vov
        self._rho = rho
        self._sigma_0 = sigma_0

    @property
    def c(self):
        return self._c
    @c.setter
    def c(self, new_c):
        self._c = new_c
    
    @property
    def K_max(self):
        return self._K_max
    @K_max.setter
    def K_max(self, new_K_max):
        self._K_max = new_K_max
        
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
        return r'c={:.3%}, K_max={:.2f}bps, vov={:.0%}, $\rho$={:.0%}, ATM={:.2%}, f={:.2%}, $\sigma_0$={:.2%}'        .format(self.c, self.K_max/ONE_BP, self.vov, self.rho, self.ATM, self.f, self.sigma_0)
    
    def local_vol_inv_int(self, x):
        """Closed form for the primitive of 1/(exp(-c*log(x/K_max)^2))
        """
        return (np.exp(-1/(4*self.c))*self.K_max*np.sqrt(np.pi)                *erfi((1+2*self.c*np.log(x/self.K_max))/(2*np.sqrt(self.c))))/(2*np.sqrt(self.c))
    
    @abc.abstractmethod
    def smile_func(self, K):
        pass

class SABR_AS_LN(SABR_AS_base_model):
    """SABR_AS model using lognormal implied vol quoting
    """
    def __init__(self, K_max=1, c=1, vov=1, rho=0,                 sigma_0=None, f=None, T_expiry=1,                 moneyness_lo=None, moneyness_hi=None, K_lo=None, K_hi=None,                 strike_type='logmoneyness', n_strikes=50):
        super().__init__(K_max=K_max, c=c, vov=vov, rho=rho,                         sigma_0=sigma_0, f=f, T_expiry=T_expiry,                         vol_type='LN',                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         K_lo=K_lo, K_hi=K_hi,                         strike_type=strike_type, n_strikes=n_strikes)
    
    @property
    def model_name(self):
        return 'SABR_AS_LN'
    
    @property
    def ATM_LN(self):
        f_adj = self.f/self.K_max
        C_f_adj = np.exp(-self.c*np.log(f_adj)**2)    
        gamma_1 = -(2*self.c*np.log(f_adj))/self.f
        gamma_2 = (2*self.c*(-1+np.log(f_adj)+2*self.c*np.log(f_adj)**2))/(self.f**2)
        
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
            f_avg_adj = f_avg/self.K_max
            C_f_avg_adj = np.exp(-self.c*np.log(f_avg_adj)**2)
            
            gamma_1 = -(2*self.c*np.log(f_avg_adj))/f_avg
            gamma_2 = (2*self.c*(-1+np.log(f_avg_adj)+2*self.c*np.log(f_avg_adj)**2))/(f_avg**2)
            
            denom = self.local_vol_inv_int(self.f)-self.local_vol_inv_int(K)
            zeta = self.vov/self.sigma_0*(self.f-K)/C_f_avg_adj
            q_zeta = 1-2*self.rho*zeta+zeta**2
            x_zeta = np.log((np.sqrt(q_zeta)-self.rho+zeta)/(1-self.rho))
            return self.sigma_0*log_moneyness/denom*zeta/x_zeta        *(1          +((2*gamma_2-gamma_1**2+1/(f_avg**2))/24*self.sigma_0**2*C_f_avg_adj**2           +0.25*self.rho*self.vov*self.sigma_0*gamma_1*C_f_avg_adj           +(2-3*self.rho**2)/24*self.vov**2)          *self.T_expiry)

class SABR_AS_N(SABR_AS_base_model):
    """SABR_AS model using lognormal implied vol quoting
    """
    def __init__(self, K_max=1, c=1, vov=1, rho=0,                 sigma_0=None, f=None, T_expiry=1,                 moneyness_lo=None, moneyness_hi=None, K_lo=None, K_hi=None,                 strike_type='logmoneyness', n_strikes=50):
        super().__init__(K_max=K_max, c=c, vov=vov, rho=rho,                         sigma_0=sigma_0, f=f, T_expiry=T_expiry,                         vol_type='N',                         moneyness_lo=moneyness_lo, moneyness_hi=moneyness_hi,                         K_lo=K_lo, K_hi=K_hi,                         strike_type=strike_type, n_strikes=n_strikes)
    
    @property
    def model_name(self):
        return 'SABR_AS_N'
    
    @property
    def ATM_N(self):
        f_adj = self.f/self.K_max
        C_f_adj = np.exp(-self.c*np.log(f_adj)**2)    
        gamma_1 = -(2*self.c*np.log(f_adj))/self.f
        gamma_2 = (2*self.c*(-1+np.log(f_adj)+2*self.c*np.log(f_adj)**2))/(self.f**2)
            
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
            f_avg = np.sqrt(self.f*K)
            f_avg_adj = f_avg/self.K_max
            C_f_avg_adj = np.exp(-self.c*np.log(f_avg_adj)**2)
            
            gamma_1 = -(2*self.c*np.log(f_avg_adj))/f_avg
            gamma_2 = (2*self.c*(-1+np.log(f_avg_adj)+2*self.c*np.log(f_avg_adj)**2))/(f_avg**2)
            
            denom = self.local_vol_inv_int(self.f)-self.local_vol_inv_int(K)
            zeta = self.vov/self.sigma_0*(self.f-K)/C_f_avg_adj
            q_zeta = 1-2*self.rho*zeta+zeta**2
            x_zeta = np.log((np.sqrt(q_zeta)-self.rho+zeta)/(1-self.rho))
            return self.sigma_0*(self.f-K)/denom*zeta/x_zeta        *(1          +((2*gamma_2-gamma_1**2)/24*self.sigma_0**2*C_f_avg_adj**2           +0.25*self.rho*self.vov*self.sigma_0*gamma_1*C_f_avg_adj           +(2-3*self.rho**2)/24*self.vov**2)          *self.T_expiry)

