#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.misc import derivative
from scipy.optimize import brentq
from functools import partial
import abc

ONE_BP = 1e-4
ONE_PCT = 1e-2

# Implied vol Black/normal quoter
# Handles vol --> price and price --> vol conversions for European calls and puts
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
    def __init__(self, 
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
        # initial forward
        self._f = f
        # maturity
        self._T_expiry = T_expiry
        
        # strike of logmoneyness
        self._strike_type = strike_type
        
        # logmoneyness/strike boundaries, depending on strike_type
        self._logmoneyness_lo = logmoneyness_lo
        self._logmoneyness_hi = logmoneyness_hi
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
    def logmoneyness_lo(self):
        if self.strike_type == 'strike':
            return self.K_lo/self.f
        else:
            return self._logmoneyness_lo
    @logmoneyness_lo.setter
    def logmoneyness_lo(self, new_logmoneyness_lo):
        if self.strike_type == 'strike':
            raise NameError('Impossible to mark logmoneyness in strike mode')
        else:
            self._logmoneyness_lo = new_logmoneyness_lo
    
    @property
    def logmoneyness_hi(self):
        if self.strike_type == 'strike':
            return self.K_hi/self.f
        else:
            return self._logmoneyness_hi
    @logmoneyness_hi.setter
    def logmoneyness_hi(self, new_logmoneyness_hi):
        if self.strike_type == 'strike':
            raise NameError('Impossible to mark logmoneyness in strike mode')
        else:
            self._logmoneyness_hi = new_logmoneyness_hi
        
    @property
    def K_lo(self):
        if self.strike_type == 'logmoneyness':
            return max(self.f * np.exp(self.logmoneyness_lo), self._K_abs_min)
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
            return min(self.f * np.exp(self.logmoneyness_hi), self._K_abs_max)
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
        if new_strike_type in ['strike', 'logmoneyness']:
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
            return np.linspace(self.logmoneyness_lo, self.logmoneyness_hi, self.n_strikes)
    
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
    def smile(self):
        """Returns a dictionary strike -> implied vol
        """
        return dict(zip(self.strike_grid,                        list(map(lambda K: self.smile_func(K), self.strike_grid))))
    
    def option_price(self, K, payoff='Call'):
        """Returns the call/put price corresponding the implied vol at strike K 
        """
        return self.IV.price_from_vol(self.smile_func(K), self.f, K, self.T_expiry,                                      payoff=payoff)
    
    @property
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