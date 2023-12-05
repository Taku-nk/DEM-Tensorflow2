
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf

sys.path.append("/home/nakagawa/mylib/PythonCode/visualize_2D")
from visualize_DEM_2D import Visualize2DFormat as VizFormat


from utils.data_classes import InputData


class LayerDNN(tf.keras.layers.Layer):
    def __init__(self, dnn_in=2, dnn_out=2, dnn_layers=[20, 20, 20]):
        super().__init__()

        self.hidden_layers = []
        for l in dnn_layers:
            dense_layer = tf.keras.layers.Dense(l, activation=None, use_bias=True)
            self.hidden_layers.append(dense_layer)
        self.dnn_out_layer = tf.keras.layers.Dense(dnn_out, activation=None, use_bias=True)


    def call(self, X):
        """
        Args:
            X: tf.tensor. Shape=(batch, training_points, 2). Batch is usually 1.
        """
        temp = X
        for l in self.hidden_layers:
            temp = l(temp)
            temp = tf.nn.relu(temp) **2

        dnn_out = self.dnn_out_layer(temp)

        return dnn_out


class ModelXtoDisp(tf.keras.Model):
    def __init__(self, dnn_in=2, dnn_out=2, dnn_layers=[20, 20, 20]):
        super().__init__()
        self.dnn = LayerDNN(dnn_in=2, dnn_out=2, dnn_layers=[20, 20, 20])

    def call(self, X):
        dnn_out = self.dnn(X)

        x, y = X[:, :, 0:1], X[:, :, 1:2]

        u =  x * dnn_out[:, :, 0:1]
        v =  (y + 1.0) * dnn_out[:, :, 1:2]

        return u, v

    






if __name__=='__main__':
    data_dir = Path('./data')

    input_data = InputData(
        internal_training_point    = data_dir/"input/internal_point_and_weight.csv",
        BC_Neumann_training_point  = data_dir/"input/Neumann_BC_point_and_weight.csv",
        BC_Neumann_traction_vector = data_dir/"input/Neumann_BC_traction_vector.csv",
        validation_point           = data_dir/"input/validation_point_and_weight.csv"
    )

    model_data = input_data.get_data() # dictionary basic shape (1, :, 1) or (1, :, 2)

    model_x_to_disp = ModelXtoDisp(dnn_in=2, dnn_out=2, dnn_layers=[20, 20, 20])


    Input = tf.keras.Input(shape=(None,2)) # Determin the input shape
    model_x_to_disp(Input) # Build model (# initialize the shape)
    model_x_to_disp.summary() 


    # prediction before training
    pred = model_x_to_disp(model_data['X_int'])
    print(pred[0].shape)
    
    quit()








































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
