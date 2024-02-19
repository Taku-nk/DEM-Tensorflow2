from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("C:\\Users\\taku\\Hiroshima-U-Master\\OneDrive - Hiroshima University\\ドキュメント\\1kouza\\MasterResearch\\PythonCode\\plot_2d")
from plot_2d import FigPlot, clip_df


#-------------------------------------------------------------------------------
# Figure Settings
#-------------------------------------------------------------------------------
fig_plot = FigPlot(figsize=(4,3), fontsize=12)

plt.rcParams["savefig.directory"] = "./"
# plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.fontset'] = 'stix' # font looks like Times
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['Times New Roman'] 

# fig_plot.ax.set_xlim(0.5, 3)
# fig_plot.ax.set_ylim(0,7)
fig_plot.ax.set_xlabel(r'$y/a$ [m]')
fig_plot.ax.set_ylabel(r'$\sigma_{xx}$ [Pa]')
# fig_plot.ax.locator_params(axis='x',nbins=6)
# fig_plot.ax.locator_params(axis='y',nbins=6)


#-------------------------------------------------------------------------------
# Plot Data
#-------------------------------------------------------------------------------


path_foo = './DEM/一様線形解/DEM_results.csv'
# df_foo = clip_df(pd.read_csv(ansys_path), min_x=hole_size, max_x=1.0, key='x')
df_foo = pd.read_csv(path_foo)
xs_foo = df_foo.loc[:, 'y'] / hole_size
vs_foo = df_foo.loc[:, 'stress_x']
fig_plot.ax.plot(xs_foo, vs_foo, label='DEM: Linear Disp', c='tab:blue', ls='-')




fig_plot.show()



