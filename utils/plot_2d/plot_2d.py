
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker

import numpy as np
import pandas as pd


class FigPlot:
    def __init__(self, figsize=(4, 3), fontsize=8, small_fac=0.8):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)

        self.fontsize = fontsize 
        self.small_fac = small_fac

        self.formatter = ticker.ScalarFormatter(useMathText=True)
        self.config()


    def config(self):
        plt.rcParams["savefig.directory"] = "/Users/taku/Desktop"
        # plt.rcParams["savefig.format"] = 'eps'
        plt.rcParams["savefig.format"] = 'svg'
        # plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams['legend.framealpha'] = 1.0
        self.ax.locator_params(axis='x',nbins=6)
        self.ax.locator_params(axis='y',nbins=6)
        self.ax.tick_params(axis='both', labelsize=self.fontsize*self.small_fac)

        # formatter = ticker.ScalarFormatter(useMathText=True)
        self.formatter.set_scientific(True)
        # formatter.set_powerlimits((-1,1))
        self.ax.yaxis.set_major_formatter(self.formatter)

        self.ax.yaxis.offsetText.set_fontsize(self.fontsize*self.small_fac)

        # self.ax.set_xlim(-5, 5)

        # self.ax.set_title("Displacement z", fontsize=16, fontname="Times New Roman")
        self.ax.set_xlabel("Coordinate $x$ [m]", fontsize=self.fontsize)
        self.ax.set_ylabel("Value $u$ [m]", fontsize=self.fontsize)
        self.ax.grid(ls=':')



    def show(self, legend=True):
        if legend==True:
            self.ax.legend(fontsize=self.fontsize*self.small_fac)
        plt.tight_layout()
        plt.show()


def clip_df(df, min_x, max_x, key='x'):
    """Clip DataFrame from min_x to max_x

    Args:
        df: pandas dataframe that contains 'x' column
        min_x: float. min clip position
        max_x: float. max clip position
        key: str. Specifies which column key to filter. Default is 'x'
    
    Returns:
        clipped_df: pd.DataFrame. clipped result dataframe
    """

    clipped_df = df.loc[(df[key]>min_x) & (df[key]<max_x)]
    return clipped_df



if __name__ == '__main__':
    from pathlib import Path

    hole_size = 1

    df = clip_df(pd.read_csv('./test_result/Ansys_sampled.csv'), min_x=1.0, max_x=5.0, key='x')
    xs = df.loc[:, 'x'] / hole_size
    vs = df.loc[:, 'stress_y']

    df_dem = clip_df(pd.read_csv('./test_result/DEM_results.csv'), min_x=1.0, max_x=5.0, key='x')
    xs_dem = df_dem.loc[:, 'x']
    vs_dem = df_dem.loc[:, 'stress_y']

    fig_plot = FigPlot(figsize=(4,3), fontsize=12)
    fig_plot.ax.set_xlim(0.5, 3)
    fig_plot.ax.set_xlabel(r'$y/R$ [m]')
    fig_plot.ax.set_ylabel(r'$\sigma_{yy}$ [Pa]')

    
    fig_plot.ax.plot(xs, vs, label='FEM', c='tab:gray', ls=':')
    fig_plot.ax.plot(xs_dem, vs_dem, label='DEM', c='tab:gray', ls='-')

    fig_plot.show()

