#!/usr/bin/env python
# coding: utf-8

# ## Import

# In[1]:


from functools import partial
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.optimize import root
import matplotlib.pyplot as plt
from functools import lru_cache
import abc


# ## Interest Rates

# In[2]:


class Rate:
    """
    For now only supports constant interest rate discounting
    """
    def __init__( self, short_rate, pricing_date = 0 ):
        self._short_rate    = short_rate
        self._security_type = "CT Rate"
        self._pricing_date  = pricing_date
        
    @property
    def short_rate( self ):
        return self._short_rate

    @property
    def pricing_date( self ):
        return self._pricing_date

    @pricing_date.setter
    def pricing_date( self, new_pricing_date):
        self._pricing_date = new_pricing_date
        
    @property
    def security_type( self ):
        return self._security_type

# for now only use constant rate
EUR_Libor = Rate( 0 );
USD_Libor = Rate( 3 / 100 );

CSA_map = {
    "EUR-Libor" : EUR_Libor,
    "USD-Libor" : USD_Libor,
}


# ## Eq Ticker Map

# In[3]:


# add more tickers here
Ticker_Map = {
    "AAPL":   "Apple"
}


# ## Eq Stock

# In[4]:


class Underlyer(abc.ABC):
    """
    Class for option underlyer
    """
    def __init__( self, spot_value, pricing_date = 0 ):
        self._spot_value    = spot_value
        self._pricing_date  = pricing_date
    
    @property
    @abc.abstractmethod
    def security_type( self ):
        pass
    
    @property
    def spot_value( self ):
        return self._spot_value
    
    @spot_value.setter
    def spot_value( self, new_spot_value):
        self._spot_value = new_spot_value
        
    @property
    def pricing_date( self ):
        return self._pricing_date

    @pricing_date.setter
    def pricing_date( self, new_pricing_date):
        self._pricing_date = new_pricing_date
    
class EqStock( Underlyer ):
    def __init__( self, spot_value, ticker, pricing_date = 0 ):
        super().__init__( spot_value, pricing_date )
        self._ticker        = ticker
        self._security_type = "Eq Stock"
        self._stock_name    = Ticker_Map.get( ticker )
    
    @property
    def security_type( self ):
        return self._security_type
    
    @property
    def ticker( self ):
        return self._ticker


# ## Eq Vol

# In[5]:


class Vol:
    """
    Generic vol class
    """
    def __init__( self, deal_terms, vol_type ):
        self._deal_terms    = deal_terms
        self._vol_type      = vol_type
        self._pricing_date  = deal_terms[ "underlyer" ].pricing_date
        
        
    def price_from_vol( self, vol ):
        """
        Black-Scholes Call/Put price from volatility, the rest of the parameters (strike, rates, maturity) 
        are provided in the deal terms
        """
        if self._vol_type == "LogNormal":
            S           = self._deal_terms[ "underlyer" ].spot_value
            K           = self._deal_terms[ "payoff" ].payoff_terms[ "strike" ]
            time_to_mat = self._deal_terms[ "maturity" ] - self._pricing_date
            r           = CSA_map[ self._deal_terms[ "CSA" ] ].short_rate
            d1          = 1 / ( vol * np.sqrt( time_to_mat ) ) * ( np.log( S / K ) + ( r + 0.5 * vol ** 2 ) * time_to_mat )
            d2          = d1 - vol * np.sqrt( time_to_mat ) 
            CallPrice   = S * norm.cdf( d1 ) - K * np.exp( -r * time_to_mat ) * norm.cdf( d2 ) 

            if self._deal_terms[ "payoff" ].payoff_name == "European Call":
               return CallPrice
            elif self._deal_terms[ "payoff" ].payoff_name == "European Put":
               return CallPrice + K * np.exp( -r * time_to_mat ) - S 
            else:
                raise NameError( "Unsupported vol type : " + self._deal_terms[ "Payoff" ].payoff_name )
        else:
            raise NameError( "Unsupported vol type : " + self._vol_type )
  
    def vol_from_price( self, price ): 
        """
        Black-Scholes Call/Put implied volatility from price, the rest of the parameters (strike, rates, maturity) 
        are provided in the deal terms
        """
        def target_func( price, vol ):
            return self.price_from_vol( vol ) - price 
                                               
        return brentq( partial( target_func, price ), 1e-8, 10 ) 
                
class EqVol( Vol ):
    """
    Equity vol class, only supports log-normal for now
    """
    def __init__( self, deal_terms ):
        super().__init__( deal_terms, "LogNormal" )
        self._security_type = "Eq Vol"
    


# ## Eq Option

# In[6]:


class Option(abc.ABC):
    """
    Generic class for options
    """
    def __init__( self, deal_terms ):
        self._deal_terms    = deal_terms
        self._vol_fit_type  = ""
        self._vol_curve     = None
        self._vol_object    = None
        self._pricing_date  = deal_terms[ "underlyer" ].pricing_date
        
    @property
    @abc.abstractmethod
    def security_type( self ):
        pass

    @property
    def deal_terms( self ):
        return self._deal_terms
    
    def price( self ):
        self.calibrate()
        
        payoff_name = self._deal_terms[ "payoff" ].payoff_name
        # only supports Call/Put for now
        if payoff_name not in [ "European Call", "European Put" ]: 
            raise NameError( "Unsupported Payoff Name : " + self._deal_terms[ "Payoff" ].payoff_name )
        elif payoff_name == "European Call":
            price_curve_func    = self._vol_curve.curve_func( rotation_type = "Call Price" ) 
        elif payoff_name == "European Put":
            price_curve_func    = self._vol_curve.curve_func( rotation_type = "Put Price" ) 
            
        strike = self._deal_terms[ "payoff" ].payoff_terms[ "strike" ];
            
        return( price_curve_func( strike ) )

    ## TODO : better handle multiple maturity, for now only one is available
    
    def precheck_calibration( self ):
        """
        Check that vol surface is avaible in the option currency and maturity
        """
        maturity = self._deal_terms[ "maturity" ]
        if self._vol_curve._slice != maturity:
            raise NameError( "Slice {} doesn't exist: cannot price option with this maturity".format(maturity) )
        
        denominated = self._deal_terms[ "denominated" ]
        if self._vol_curve._denominated != denominated:
            raise NameError( "Vol curve for {} doesn't exist: cannot price option denominated in this currency".format(denominated) )

        CSA = self._deal_terms[ "CSA" ]
        if self._vol_curve._CSA != CSA:
            raise NameError( "Vol curve for {} CSA doesn't exist: cannot price option with this CSA".format(CSA) )
        
        ticker = self._deal_terms[ "underlyer" ].ticker
        if self._vol_curve._underlyer.ticker != ticker:
            raise NameError( "Vol curve for {} doesn't exist: cannot price option on this ticker".format(ticker) )
            
    # cache calibration
    @lru_cache(maxsize=None)
    def calibrate( self ):
        self.precheck_calibration()
        self._vol_curve.curve_fit()
            
class EqOption( Option ):
    """
    Class for equity options
    """
    def __init__( self, deal_terms, Mkt = None ):
        super().__init__( deal_terms )
        self._security_type = "Eq Option"
        self._vol_object    = EqVol( deal_terms )
        self._Mkt           = Mkt
        # EqOption only supports EqStock underlyer
        if self._deal_terms[ "underlyer" ].security_type != "Eq Stock" :
            raise NameError( "Unsupported Security Type : " + self._deal_terms[ "underlyer" ].security_type )
        
        self._vol_fit_type = "SVI" 
        self._vol_curve   = EqVolCurve( deal_terms[ "underlyer" ], self._vol_fit_type, Mkt = Mkt )
    
    @property
    def security_type( self ):
        return self._security_type
        
class EqVolConstraint( Option ):
    """
    Class for market observable options used to build equity vol surfaces
    """
    def __init__( self, deal_terms, mark, mark_type ):
        super().__init__( deal_terms )
        self._security_type     = "Eq Vol Constraint"
        self._vol_object        = EqVol( deal_terms )
        
        # EqOption only supports EqStock underlyer
        if self._deal_terms[ "underlyer" ].security_type != "Eq Stock":
            raise NameError( "Unsupported Security Type : " + self._deal_terms[ "underlyer" ].security_type )
        else:
            self._underlier = self._deal_terms[ "underlyer" ] 
            
        # EqVolConstraint only supports European Call payoff
        # Pricing model (PM) is BS_simple
        # Unused for now, leave it here for later if different pricing model are implemented
        if self._deal_terms[ "payoff" ].payoff_name in [ "European Call", "European Put" ]: 
            self._PMName = "BS_simple"
        else:
            raise NameError( "Unsupported Payoff Name : " + self._deal_terms[ "Payoff" ].payoff_name )

        if mark_type not in [ "price", "LN vol" ]:
            raise NameError( "Unsupported marking convention : " + mark_type )
        
        self._mark      = mark
        self._mark_type = mark_type
        
        if self._mark_type == "price":
            self._price         = mark
            self._implied_vol   = self._vol_object.vol_from_price( mark )
        elif self._mark_type == "LN vol":
            self._price         = self._vol_object.price_from_vol( mark )
            self._implied_vol   = mark
            
        self._strike    = self._deal_terms[ "payoff" ].payoff_terms[ "strike" ]
        self._is_ATM    = self._strike == self._underlier.spot_value
    
    @property
    def strike( self ):
        return self._strike
    
    @property
    def is_ATM( self ):
        return self._is_ATM
    
    @property
    def price( self ):
        if self._mark_type == "price":
            return self._mark
        elif self._mark_type == "LN Vol":
            return self._vol_object.price_from_vol( self._mark )
    
    @property
    def implied_vol( self ):
        if self._mark_type == "price":
            return self._vol_object.vol_from_price( self._mark )
        elif self._mark_type == "LN Vol":
            return self._mark
        
    @property
    def security_type( self ):
        return self._security_type


# In[7]:


class Payoff(abc.ABC):
    """
    Generic class for payoff function
    """
    def __init__( self, payoff_name, payoff_terms ):
        self._payoff_name  = payoff_name
        self._payoff_terms = payoff_terms 
        
    @property
    @abc.abstractmethod
    def payoff_name( self ):
        pass
        
    @property
    @abc.abstractmethod
    def payoff_terms( self ):
        pass
    
class EuropeanCall( Payoff ):
    def __init__( self, strike ):
        super().__init__( "European Call", { "strike" : strike } )
    
    @property
    def payoff_name( self ):
        return self._payoff_name
        
    @property
    def payoff_terms( self ):
        return self._payoff_terms
        
class EuropeanPut( Payoff ):
    def __init__( self, strike ):
        super().__init__( "European Put", { "strike" : strike } )

    @property
    def payoff_name( self ):
        return self._payoff_name
        
    @property
    def payoff_terms( self ):
        return self._payoff_terms
    
class EuropeanFormula( Payoff ):
    def __init__( self, formula ):
        super().__init__( "European Formula", { "formula" : formula } )
    
    @property
    def payoff_name( self ):
        return self._payoff_name
        
    @property
    def payoff_terms( self ):
        return self._payoff_terms


# ## Eq Vol Curve
# Implemets slice-by-slice vol fitting à la Gatheral's SVI
# 
# Raw SVI : 
# w(k) = a + b * ( rho * ( k - m ) + sqrt( ( k - m ) ^ 2 + sigma ^ 2 ) )

# In[8]:


class VolCurve:
    """
    Generic class for volatility curve
    """
    def __init__( self, mkt_instruments, fit_type ):
        self._mkt_instruments   = mkt_instruments
        self._fit_type          = fit_type
        self._fit_info          = None
        self._parms             = []
        self._is_fitted         = False
        self._curve_func        = None
        
        # Parameters for plot
        self._n_plot            = 100
        self._K_lo              = 50
        self._K_hi              = 150
        
        [ underlyer_set, CSA_set, maturity_set, denominated_set ] = [ set(), set(), set(), set() ]
        for mkt_instrument in mkt_instruments:
            underlyer_set.add( mkt_instrument._deal_terms[ "underlyer" ] )
            CSA_set.add( mkt_instrument._deal_terms[ "CSA" ] )
            maturity_set.add( mkt_instrument._deal_terms[ "maturity" ] )
            denominated_set.add( mkt_instrument._deal_terms[ "denominated" ] )
            
        if len( underlyer_set ) > 1:
            raise NameError( "More than 1 underlyer!" )
        else:
            self._underlyer = list( underlyer_set )[ 0 ]
        
        if len( CSA_set ) > 1:
            raise NameError( "More than 1 CSA!" )
        else:
            self._CSA = list( CSA_set )[ 0 ]
            
        if len( maturity_set ) > 1:
            raise NameError( "More than 1 slice!" )
        else:
            self._slice = list( maturity_set )[ 0 ]
        
        if len( denominated_set ) > 1:
            raise NameError( "More than 1 currency!" )
        else:
            self._denominated = list( denominated_set )[ 0 ]
        
        self._pricing_date  = self._underlyer.pricing_date
        
    @property
    @abc.abstractmethod
    def security_type( self ):
        pass

    @property
    def n_plot( self ):
        return self._n_plot

    @n_plot.setter
    def n_plot( self, new_n_plot):
        self._n_plot = new_n_plot
        
    @property
    def K_hi( self ):
        return self._K_hi
    
    @K_hi.setter
    def K_hi( self, new_K_hi):
        self._K_hi = new_K_hi
        
    @property
    def K_lo( self ):
        return self._K_lo
    
    @K_lo.setter
    def K_lo( self, new_K_lo):
        self._K_lo = new_K_lo
    
    @property
    def parms( self ):
        return self._parms
    
    @parms.setter
    def parms( self, new_parms):
        self._parms     = new_parms
        self._is_fitted = True
    
    @property
    def is_fitted( self ):
        return self._is_fitted
    
    @is_fitted.setter
    def is_fitted( self, new_is_fitted):
        self._is_fitted     = new_is_fitted
        
    def SVI_curve_func( self, rotation_type = "Total Variance" ):
        S = self._underlyer.spot_value
        F = S * np.exp( CSA_map[ self._CSA ].short_rate * self._slice )

        def total_variance_func( parms, K ):
            # K : strike
            # k : log-moneyness
            k      = np.log( K / F )
            a      = parms[ 0 ]
            b      = parms[ 1 ]
            rho    = parms[ 2 ]
            m      = parms[ 3 ]
            sigma  = parms[ 4 ]
            # this is w(k.t) where k : log-moneyness, t : slice (maturity)
            # w(k,t) = sigma_bs(k,t)^2 * t total variance
            return( a + b * ( rho * ( k - m ) + np.sqrt( ( k - m ) ** 2 + sigma ** 2 ) ) )
        
        if rotation_type == "Total Variance":
            return( total_variance_func )
        elif rotation_type == "Variance":
            def variance_func( parms, K ):
                return( total_variance_func( parms, K ) / self._slice )
            return( variance_func )
        elif rotation_type == "Vol":
            def vol_func( parms, K ):
                return( np.sqrt( total_variance_func( parms, K ) / self._slice ) )
            return( vol_func )
        elif rotation_type in [ "Call Price", "Put Price" ]:
            def price_func( parms, K ):
                DT = {
                    "underlyer"      : self._underlyer,
                    "maturity"       : self._slice,
                    "denominated"    : self._denominated,
                    "CSA"            : self._CSA
                    }
                
                if rotation_type == "Call Price":
                    DT[ "payoff"] = EuropeanCall( strike = K )
                elif rotation_type == "Put Price":
                    DT[ "payoff"] = EuropeanPut( strike = K )
                else: 
                    raise NameError( "Unknown payoff" )
                    
                vol_at_strike_obj   = EqVol( DT )
                implied_vol         = np.sqrt( total_variance_func( parms, K ) / self._slice )
                return( vol_at_strike_obj.price_from_vol( implied_vol ) )
            return( price_func )

    def constraints_func( self ): 
        if self._fit_type == "SVI":
            func        = self.SVI_curve_func( rotation_type = "Total Variance")
            init_parms  = [ 0, 0.1, 0.5, 0.5, 0.2 ]
        else:
            return None
        
        def all_constraints_func( parms ):
            constr = []
            for mkt_instrument in self._mkt_instruments:
                constr.append(  func( parms, mkt_instrument._strike ) 
                                - mkt_instrument._implied_vol ** 2 * self._slice ) 
            return( constr )
        
        return( { "func" : all_constraints_func, "init_parms" : init_parms } )
    
    def curve_func( self, rotation_type = "Total Variance" ): 
        if not self._is_fitted:
            return None
        
        if self._fit_type == "SVI":
            func = self.SVI_curve_func( rotation_type )
        else: 
            return None
        
        return partial( func, self._parms )
    
    def curve_fit( self, force_fit = False ):       
        # to add : constrained optimization to take care of positivity condition
        # e.g : a + b * sigma * sqrt( 1 - rho ^ 2 ) >= 0
        if not self._is_fitted or force_fit:
            constraints         = self.constraints_func()
        
            sol                 = root( constraints[ "func" ], constraints[ "init_parms" ], method = 'lm')   
            self._fit_info      = sol
            self._parms         = sol.x
            self._is_fitted     = True 
            self._curve_func    = self.curve_func( rotation_type = "Total Variance") 
        
    def plot( self ):
        if self._is_fitted:
            strikes             = np.linspace( self._K_lo, self._K_hi, self._n_plot )
            vols                = []
            prices              = []
            vol_curve_func      = self.curve_func( "Vol" )
            price_curve_func    = self.curve_func( "Call Price" )

            for K in strikes:
                vols.append( vol_curve_func( K ) )
                prices.append( price_curve_func( K ) )

            mkt_strikes = []
            mkt_vols    = []
            mkt_prices  = []

            for mkt_instrument in self._mkt_instruments:
                mkt_strikes.append( mkt_instrument.strike )
                mkt_vols.append( mkt_instrument.implied_vol )
                mkt_prices.append( mkt_instrument.price )

            #plt.plot( strikes, vols )
            #plt.plot( mkt_strikes, mkt_vols, "ro" )
            #plt.plot( mkt_strikes, mkt_prices, "bo" )

            f, (ax1, ax2) = plt.subplots(2, figsize=(8, 7))
            
            ax1.plot(mkt_strikes, mkt_vols, "ro", label="Market vols")
            ax1.plot(strikes, vols, label="Fitted vol curve")

            ax1.set_xlabel('Strike')
            ax1.set_ylabel('Implied volatility')
            ax1.legend()

            ax2.plot(mkt_strikes, mkt_prices, "ro", label="Market prices")
            ax2.plot(strikes, prices, label="Fitted price curve")

            ax2.set_xlabel('Strike')
            ax2.set_ylabel('Call Price')
            ax2.legend()

            plt.tight_layout()
            plt.show()
            
            
            #a      = parms[ 0 ]
            #b      = parms[ 1 ]
            #rho    = parms[ 2 ]
            #m      = parms[ 3 ]
            #sigma  = parms[ 4 ]
            
            print('w(k) = a + b * ( rho * ( k - m ) + sqrt( ( k - m ) ^ 2 + sigma ^ 2 ) )')
            print('k : log-moneyness, t : slice (maturity), w(k,t)=sigma_bs(k,t)^2 * t (total variance)')
            print('a={:.4f}\nb={:.4f}\nrho={:.4f}\nm={:.4f}\nsigma={:.4f}'.format(self._parms[0], self._parms[1], self._parms[2], self._parms[3], self._parms[4]))

class EqVolCurve( VolCurve ):
    def __init__( self, eq_underlyer, fit_type, Mkt = None ):
        if fit_type not in [ "SVI" ]:    
            raise NameError( "Unsupported fit type : " + fit_type )
            
        mkt_instruments = Mkt.Mkt[ eq_underlyer.ticker ]
        
        super().__init__( mkt_instruments, fit_type )
        self._security_type = "Eq Vol Curve"

    @property
    def security_type( self ):
        return self._security_type


# # Observable option market

# In[9]:


class VolMkt( abc.ABC ):
    """
    Generic class to store observable options
    """
    def __init__( self, Mkt ):
        self._Mkt = Mkt
        
    @property
    @abc.abstractmethod
    def security_type( self ):
        pass

class EqVolMkt( VolMkt ):
    """
    Class to store observable eq options
    """
    def __init__( self, Mkt ):
        super().__init__( Mkt )
        self._security_type = "Eq Vol Mkt"
        
    @property
    def security_type( self ):
        return self._security_type

    @property
    def Mkt( self ):
        return self._Mkt
    
    @Mkt.setter
    def Mkt( self, new_Mkt ):
        self._Mkt = new_Mkt

