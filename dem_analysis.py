
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append("/home/nakagawa/mylib/PythonCode/visualize_2D")
from visualize_DEM_2D import Visualize2DFormat as VizFormat


from utils.data_classes import InputData



if __name__=='__main__':
    data_dir = Path('./data')

    input_data = InputData(
        internal_training_point    = data_dir/"input/internal_point_and_weight.csv",
        BC_Neumann_training_point  = data_dir/"input/Neumann_BC_point_and_weight.csv",
        BC_Neumann_traction_vector = data_dir/"input/Neumann_BC_traction_vector.csv",
        validation_point           = data_dir/"input/validation_point_and_weight.csv"
    )

    model_data = input_data.get_data() # dictionary basic shape (1, :, 1) or (1, :, 2)










































    # ------------------------------------------------------------------------------
    # Script starts
    # ------------------------------------------------------------------------------

    # Format
    #-------------------------------------------------------------------------------
    viz_format = VizFormat(figsize = (4, 3), fontsize=12)

    plt.rcParams["savefig.directory"] = "./"
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.fontset'] = 'stix'
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman']
    # plt.rcParams['font.sans-serif'] = ['Times New Roman']

    viz_format.ax.set_xlabel(r"$x$")
    viz_format.ax.set_ylabel(r"$y$")

    # viz_format.ax.set_xlim(0.2, 0.4)
    # viz_format.ax.set_ylim(0, 0.2)



    xs  = model_data['X_int'][0, :, 0]
    ys  = model_data['X_int'][0, :, 1]
    val = model_data['X_int'][0, :, 0]

    viz_format.plot_mesh(xs, ys, val, contour_num=40, contour_line = False, cmap='turbo')
