#!/usr/bin/env python
# coding: utf-8

import os
import time
import pandas as pd
from pandas.compat import StringIO
from zipfile import ZipFile
from datetime import datetime

class Reader():
    def __init__(self,
                 fname: str='',
                 price_mult: float=10000.0,
                 level: int=1,
                ) -> None:
        self._fname = fname # filename
        self._price_mult  = price_mult # scaling factor for price
        self._level  = level # load up to this orderbook depth
        self._df = pd.DataFrame()
        
    @property
    def df(self) -> pd.DataFrame:
        return self._df
        
    @property
    def fname(self) -> str:
        return self._fname
    @fname.setter
    def fname(self, new_fname) -> None:
        self._fname = new_fname
    
    @property
    def price_mult(self) -> float:
        return self._price_mult
    @price_mult.setter
    def price_mult(self, new_price_mult) -> None:
        self._price_mult = new_price_mult
        
    @property
    def level(self) -> int:
        return self._level
    @level.setter
    def level(self, new_level) -> None:
        self._level = new_level
        
    def orderbook_single_load(self, 
                              fn: str,
                              zf: ZipFile,
                             ) -> pd.DataFrame:
        """
        Ask Price 1:  Level 1 ask price  (best ask price)
        Ask Size 1:  Level 1 ask volume  (best ask volume)
        Bid Price 1:  Level 1 bid price  (best bid price)
        Bid Size 1:  Level 1 bid volume  (best bid volume)
        Ask Price 2:  Level 2 ask price  (second best ask price)
        Ask Size 2:  Level 2 ask volume (second best ask volume)
        """
        columns={}
        for i in range(self.level):
            columns[4*i+0] = 'Ask Price {}'.format(i+1)
            columns[4*i+1] = 'Ask Size {}'.format(i+1)
            columns[4*i+2] = 'Bid Price {}'.format(i+1)
            columns[4*i+3] = 'Bid Size {}'.format(i+1)

        data = zf.open(fn).read().decode()
        df = pd.read_csv(StringIO(data), header=None, usecols=list(range(4*self.level)))
        df = df.rename(columns=columns)
        for i in range(1, self.level+1):
            df['Ask Price {}'.format(i)] = df['Ask Price {}'.format(i)] / self.price_mult
            df['Bid Price {}'.format(i)] = df['Bid Price {}'.format(i)] / self.price_mult
            df['Mid Price {}'.format(i)] = 0.5*(df['Bid Price {}'.format(i)] + df['Bid Price {}'.format(i)])
                                                
        return df
    
    def message_single_load(self, 
                            fn: str,
                            zf: ZipFile,
                            dt: datetime,
                           ) -> pd.DataFrame:
        """
        Time (sec) : second from start of day as defined by dt
        Event Type
        Order ID
        Size
        Price
        Direction
        Add 'Time' : Time (sec) + seconds from epoch --> datetime
        """
        # columns have to be a dictionary
        columns = ['Time (sec)', 'Event Type', 'Order ID', 'Size', 'Price', 'Direction']
        columns = dict(zip(range(len(columns)), columns))

        data = zf.open(fn).read().decode()
        df = pd.read_csv(StringIO(data), header=None, usecols=list(range(len(columns))))
        df = df.rename(columns=columns)
                         
        # convert time from seconds elapsed from start of day to actual time 
        dt = datetime.strptime(fn.split('_')[1], '%Y-%m-%d')
        day_sec_since_epoch = time.mktime(dt.timetuple()) + dt.microsecond/1000000.0
        df['Time'] = df['Time (sec)'].apply(
            lambda x: datetime.fromtimestamp(x + day_sec_since_epoch)
        )
        return df
    
    def load(self, 
             start_date: datetime, 
             end_date: datetime
            ) -> None:
        """Lobster data files come by two :
        1) Orderbook : snapshot of bid/ask price/size at various levels
        2) Messages : events executed on the orderbook
        
        Messages come with a timestamp, and every row in the orderbook file corresponds to an
        event in the message file.
        
        File names are 'Ticker_Date_xxx' and correspond to a full day of trading.
        Only files with Date between start_date and end_date are loaded.
        
        This function loads prices and sizes from those orderbooks, up to a self.level depth. 
        In addition, each price change is stamped with the corresponding event time.
        """
        dfs = []
        dfs_messages = []
        with ZipFile(self.fname, 'r') as zf:
            lst = zf.namelist()
            for fn in lst:
                fn_low = fn.lower()
                # date
                dt = datetime.strptime(fn.split('_')[1], '%Y-%m-%d')
                
                if dt >= start_date and dt <= end_date:
                    if 'message' in fn_low:
                        dfs_messages.append(self.message_single_load(fn, zf, dt))
                    elif 'orderbook' in fn_low:
                        dfs.append(self.orderbook_single_load(fn, zf))

        df = pd.concat(dfs)
        df_messages = pd.concat(dfs_messages)
        
        # use 'Time' as index for the price moves
        df['Time'] = df_messages['Time']
        df.index = df['Time']
        self._df = df
