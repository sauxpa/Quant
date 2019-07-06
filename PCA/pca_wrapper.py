#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.decomposition import PCA
from functools import lru_cache
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.style.use('dark_background')


# In[2]:


class PCA_wrapper():
    """Wrapper class for PCA analysis
    """
    
    def __init__(self, df=None, n_components=3):
        self._df = df
        self._n_components = n_components

    @property
    def df(self):
        return self._df
    @df.setter
    def df(self, new_df):
        type(self).pca.fget.cache_clear()
        type(self).transformed_data.fget.cache_clear()
        self._df = new_df
    
    @property
    def n_components(self):
        return self._n_components
    @n_components.setter
    def n_components(self, new_n_components):
        type(self).pca.fget.cache_clear()
        type(self).transformed_data.fget.cache_clear()
        self._n_components = new_n_components
    
    @property
    @lru_cache(maxsize=None)
    def pca(self):
        if type(self.df) != type(None):
            pca = PCA(n_components=self.n_components)
            pca.fit(self.df)
            return pca

    @property
    @lru_cache(maxsize=None)
    def transformed_data(self):
        if type(self.df) != type(None):
            return self.pca.transform(self.df)
    
    @property
    def explained_variance_ratio_(self):
        return self.pca.explained_variance_ratio_
    
    def summary_variance(self):
        evrs = self.explained_variance_ratio_
        for i, evr in zip(range(self.n_components), evrs):
            print('Component {} : {:.2%}'.format(i+1, evr))
        print('Total explained variance : {:.2%}'.format(sum(evrs)))
        
    def plot_pca_2d_population(self, i=1, j=2):
        """Plot the projection of individual datapoints in the population on
        components i and j
        """
        fig, ax = plt.subplots(nrows=1, ncols=1)
        data = self.transformed_data
        colormap = plt.cm.coolwarm
        im = ax.scatter(data[:,i-1], data[:,j-1], c=range(data.shape[0]), cmap=colormap)
        ax.set_xlabel('Component {}'.format(i))
        ax.set_ylabel('Component {}'.format(j))
        
        years = sorted(list(set([x.year for x in self.df.index])))
        ticks = np.linspace(0, data.shape[0]-1, len(years))
        cbar = fig.colorbar(im, ticks=ticks, orientation='vertical')
        cbar.ax.set_yticklabels(years)
        
        plt.tight_layout()
        plt.show()

    def plot_pca_2d_variables(self, i=1, j=2):
        """Plot the representation of the variables against components i and j
        """
        fig, ax = plt.subplots(nrows=1, ncols=1)
        data = self.pca.components_
        ax.scatter(data[i-1,:], data[j-1,:])
        ax.set_xlabel('Component {}'.format(i))
        ax.set_ylabel('Component {}'.format(j))

        for k, txt in enumerate(self.df.columns):
            ax.annotate(txt, (data[i-1, k], data[j-1, k]))
            
        plt.tight_layout()
        plt.show()

