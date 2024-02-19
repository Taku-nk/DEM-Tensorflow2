import sys
sys.path.append("/home/nakagawa/mylib/PythonCode/plot_2d")
from plot_2d import FigPlot, clip_df

from pathlib import Path
import numpy as np
import pandas as pd




df = clip_df(pd.read_csv('./Ansys_sampled.csv'), min_x=1.0, max_x=5.0, key='x')
xs = df.loc[:, 'x']
vs = df.loc[:, 'stress_y']


fig_plot = FigPlot(figsize=(4,3), fontsize=12)
fig_plot.ax.set_xlim(0.5, 3)
fig_plot.ax.set_xlabel(r'$x/R$ [m]')
fig_plot.ax.set_ylabel(r'$\sigma_{yy}$ [Pa]')


fig_plot.ax.plot(xs, vs, label='FEM', c='tab:gray', ls=':')

fig_plot.show()
