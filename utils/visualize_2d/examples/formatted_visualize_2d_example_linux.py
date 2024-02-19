""" 2D mesh contour visualization example for linux

This script shows the 2D scattered, non-grid data.
The data shape as to be xs, ys, vs. Usage is similar to plt.scatter(xs, ys, c=vs)
"""
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("/home/nakagawa/mylib/PythonCode/visualize_2D")
from visualize_DEM_2D import Visualize2DFormat as VizFormat


import pandas as pd

# ------------------------------------------------------------------------------
# Script starts
# ------------------------------------------------------------------------------

# Format
#-------------------------------------------------------------------------------
viz_format = VizFormat(figsize = (4, 3), fontsize=12)

plt.rcParams["savefig.directory"] = "./"
# plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['Times New Roman'] 

viz_format.ax.set_xlabel(r"$x$", math_fontfamily='cm')
viz_format.ax.set_ylabel(r"$y$", math_fontfamily='cm')

viz_format.ax.set_xlim(0.2, 0.4)
viz_format.ax.set_ylim(0, 0.2)



# Files
#-------------------------------------------------------------------------------
file_path = './Ansys.csv'
df = pd.read_csv(file_path)

# Available columnes:
# print(df.columns) -> ['x', 'y', 'disp_x', 'disp_y', 'stress_x', 'stress_y']
xs = df.loc[:, 'x'].to_numpy()
ys = df.loc[:, 'y'].to_numpy()
val = df.loc[:, 'stress_y'].to_numpy()

viz_format.plot_mesh(xs, ys, val, contour_num=20, contour_line = False, cmap='viridis')
