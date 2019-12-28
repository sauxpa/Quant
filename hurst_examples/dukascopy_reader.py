#!/usr/bin/env python
# coding: utf-8

import os
import time
import pandas as pd
from io import StringIO
from zipfile import ZipFile
from datetime import datetime

class Reader():
    def __init__(self,
                 fname: str='',
                ) -> None:
        self._fname = fname # filename
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

    def orderbook_single_load(self,
                              fn: str,
                              zf: ZipFile,
                             ) -> pd.DataFrame:

        data = zf.open(fn).read().decode()
        df = pd.read_csv(StringIO(data), header=0, usecols=list(range(5)))
        df = df.rename(columns={'Local time': 'Time'})
        df['Mid'] = 0.5*(df['Ask'] + df['Bid'])
        # transform time from str to datetime
        df['Time'] = df['Time'].transform(lambda x: datetime.strptime(x.split(' GMT')[0], '%d.%m.%Y %H:%M:%S.%f'))
        return df

    def load(self,
             start_date: datetime,
             end_date: datetime
            ) -> None:
        dfs = []
        with ZipFile(self.fname, 'r') as zf:
            lst = zf.namelist()
            for fn in lst:
                fn_low = fn.lower()
                # date
                dt = datetime.strptime(fn.split('_')[2].split('-')[0], '%d.%m.%Y')

                if dt >= start_date and dt <= end_date:
                    dfs.append(self.orderbook_single_load(fn, zf))

        df = pd.concat(dfs)

        # use 'Time' as index for the price moves
        df.index = df['Time']
        # reorder columns
        df = df.reindex(sorted(df.columns), axis=1)
        self._df = df