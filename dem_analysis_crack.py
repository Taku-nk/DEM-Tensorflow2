
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from loss_utils import loss_history_to_csv
from utils.data_classes import InputData
from dem_classes import LayerDNN, ModelXToResult, ModelDEM, LossDEM, AnalysisDEM


sys.path.append("/home/nakagawa/mylib/PythonCode/visualize_2D")
from visualize_DEM_2D import Visualize2DFormat as VizFormat

sys.path.append(
    "/home/nakagawa/mylib/PythonCode/ReadAnsysResult")
from sample_result import ResultSampler



@tf.function
def segment_adf(x1, y1, x2, y2, x, y):
    """Generate approximate distance function tensor graph (ADF) for given segment.
    Args:
        x1: float. The x position of the starting point in the segment.
        y1: float. The y position of the starting point in the segment.
        x2: float. The x position of the end point in the segment.
        y2: float. The y position of the end point in the segment.
        x: np.array. Shape=(:, 1). 
        y: np.array. Shape=(:, 1).
        
    Returns:
        phi: tf.tensor. Shape=(:, 1). Appriximate function.
    """
    vec_x = tf.concat([x, y], -1)
    vec_x1 = tf.constant([[x1, y1]])[tf.newaxis, :, :] # shape = (1, :, 2)
    vec_x2 = tf.constant([[x2, y2]])[tf.newaxis, :, :] # shape = (1, :, 2)

    vec_xc = (vec_x1 + vec_x2) / 2.0 # shape = (batch, 2) = (1, 2)

    L = tf.norm(vec_x2 -vec_x1, ord=2, axis=-1, keepdims=True)

    
    f = ((x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)) / L
    t = 1/L * ((L/2)**2 - tf.reduce_sum((vec_x - vec_xc)**2, axis=-1, keepdims=True))

    varphi = tf.sqrt(t**2 + f**4)

    # this is the apprximate distance function
    phi = tf.sqrt(f**2 + ((varphi - t) / 2)**2)

    return phi



class LayerXtoDisp(tf.keras.layers.Layer):
    def __init__(self, dnn_in=2, dnn_out=2, dnn_layers=[20, 20, 20]):
        super(LayerXtoDisp, self).__init__()
        self.dnn = LayerDNN(dnn_in=dnn_in, dnn_out=dnn_out, dnn_layers=dnn_layers)

    def call(self, X):
        dnn_out = self.dnn(X)
        zeta1 = dnn_out[:, :, 0:1]
        zeta2 = dnn_out[:, :, 1:2]
        zeta3 = dnn_out[:, :, 2:3]

        x, y = X[:, :, 0:1], X[:, :, 1:2]

        x1, y1 = (0.6, 0.0)
        x2, y2 = (1.4, 0.0)

        # How to use ADF and grad
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            g.watch(y)
            phi = segment_adf(x1, y1, x2, y2, x, y)
        phi_y = g.gradient(phi, y)
        del g


        u = x * zeta1
        v = (zeta2*phi_y + zeta3) * (y + 1)

        return u, v



if __name__=='__main__':
    # --------------------------------------
    # Prepare input
    # --------------------------------------
    data_dir = Path('./data')

    input_data_obj = InputData(
        internal_training_point    = data_dir/"input/internal_point_and_weight.csv",
        BC_Neumann_training_point  = data_dir/"input/Neumann_BC_point_and_weight.csv",
        BC_Neumann_traction_vector = data_dir/"input/Neumann_BC_traction_vector.csv",
        validation_point           = data_dir/"input/validation_point_and_weight.csv"
    )

    input_data, validation_data = input_data_obj.get_data() # dictionary basic shape (1, :, 1) or (1, :, 2)

    # Layer which contains DNN and preori boundary condition
    layer_x_to_disp = LayerXtoDisp(dnn_in=2, dnn_out=3, dnn_layers=[50, 50, 50, 50])

    # -------------------------------------
    # DEM analysis
    # -------------------------------------
    analysis_dem = AnalysisDEM(layer_x_to_disp, E=1.0, nu=0.3)
    analysis_dem.train(input_data=input_data, epochs_adam=100, epoch_sgd=30)

    analysis_dem.save_history(save_dir=data_dir/'output')



    # --------------------------------------
    # Validation
    # --------------------------------------
    validation = analysis_dem.predict(validation_data['X_val'])

    # Save result
    sampler = ResultSampler()
    sampler.load_numpy_result(
        validation_data['X_val'][0, :, 0], 
        validation_data['X_val'][0, :, 1], 
        value_dict={
            'disp_x'   : validation['disp_x'][0, :, 0].flatten(),
            'disp_y'   : validation['disp_y'][0, :, 0].flatten(),
            'stress_x' : validation['stress_x'][0, :, 0].flatten(),
            'stress_y' : validation['stress_y'][0, :, 0].flatten(),
            'stress_xy': validation['stress_xy'][0, :, 0].flatten(),
        })
    sampler.save_original(data_dir/'output/dem_result.csv')

    print("finished")
