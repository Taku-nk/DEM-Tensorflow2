""" 2D mesh contour visualization example

This script shows the 2D scattered, non-grid data.
The data shape as to be xs, ys, vs. Usage is similar to plt.scatter(xs, ys, c=vs)
"""
import matplotlib.pyplot as plt
import sys
# sys.path.append("C:\\Users\\taku\\Hiroshima-U-Master\\OneDrive - Hiroshima University\\ドキュメント\\1kouza\\MasterResearch\\PythonCode\\ReadExcel2D")
# from VizExcel import ExcelViewer, PlotExcel
sys.path.append("C:\\Users\\taku\\Hiroshima-U-Master\\OneDrive - Hiroshima University\\ドキュメント\\1kouza\\MasterResearch\\PythonCode\\visualize_2D")
from visualize_DEM_2D import debug_visualize_mesh as viz

import pandas as pd

# ------------------------------------------------------------------------------
# Script starts
# ------------------------------------------------------------------------------
file_path = './Ansys.csv'
df = pd.read_csv(file_path)


# Available columnes:
# print(df.columns) -> ['x', 'y', 'disp_x', 'disp_y', 'stress_x', 'stress_y']
xs = df.loc[:, 'x'].to_numpy()
ys = df.loc[:, 'y'].to_numpy()
val = df.loc[:, 'stress_y'].to_numpy()

viz(xs, ys, val, contour_num=50)
