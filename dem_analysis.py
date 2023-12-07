
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from loss_utils import loss_history_to_csv
from utils.data_classes import InputData
from dem_classes import LayerDNN, ModelXToResult, ModelDEM, LossDEM


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

        x, y = X[:, :, 0:1], X[:, :, 1:2]

        # How to use ADF and grad
        # u =  x * dnn_out[:, :, 0:1]
        # with tf.GradientTape(persistent=True) as g:
        #     g.watch(x)
        #     g.watch(y)
        #     phi = segment_adf(0.6, 0.0, 1.4, 0.0, x, y)
        # del g


        u =  x * dnn_out[:, :, 0:1]
        v =  (y + 1.0) * dnn_out[:, :, 1:2]

        return u, v




if __name__=='__main__':
    data_dir = Path('./data')

    input_data_obj = InputData(
        internal_training_point    = data_dir/"input/internal_point_and_weight.csv",
        BC_Neumann_training_point  = data_dir/"input/Neumann_BC_point_and_weight.csv",
        BC_Neumann_traction_vector = data_dir/"input/Neumann_BC_traction_vector.csv",
        validation_point           = data_dir/"input/validation_point_and_weight.csv"
    )

    input_data, validation_data = input_data_obj.get_data() # dictionary basic shape (1, :, 1) or (1, :, 2)

    # model_x_to_disp = ModelXtoDisp(dnn_in=2, dnn_out=2, dnn_layers=[20, 20, 20])
    layer_x_to_disp = LayerXtoDisp(dnn_in=2, dnn_out=2, dnn_layers=[20, 20, 20])
    model_x_to_result = ModelXToResult(layer_x_to_disp)


    # Input = tf.keras.Input(shape=(None,2)) # Determin the input shape
    # model_x_to_result(Input) # Build model (# initialize the shape)
    # model_x_to_result.summary() 



    # prediction before training
    # pred = model_x_to_result(model_data['X_int'])
    # print(pred['stress_x'].shape)

    model_dem = ModelDEM(model_x_to_result)
    # build
    model_dem({
        'X_int'    : tf.keras.Input(shape=(None, 2)),
        'wt_int'   : tf.keras.Input(shape=(None, 1)),
        'X_bnd'    : tf.keras.Input(shape=(None, 2)),
        'wt_bnd'   : tf.keras.Input(shape=(None, 1)),
        'Trac_bnd' : tf.keras.Input(shape=(None, 2)),
    })
  
    model_dem.summary()
    # pred_energy = model_dem(input_data) # predict 'total_energy', 'internal_energy', 'external_energy'
    # total_energy = pred_energy['total_energy']


    loss_obj = LossDEM()

    
    optimizer = tf.keras.optimizers.Adam()

    model_dem.compile(
        optimizer=optimizer,
        loss=loss_obj
    )

    @tf.function
    def train_step(input_data):
        dummy_label = np.array(0.0, dtype=np.float32)
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            pred = model_dem(input_data, training=True)
            loss = loss_obj(dummy_label, pred['total_energy'])
        gradients = tape.gradient(loss, model_dem.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model_dem.trainable_variables))


    # model
    #---------------------------------------------------------------------------
    # Train loop
    #---------------------------------------------------------------------------
    EPOCHS = 250
    loss_history = []


    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        # train_loss.reset_states()
        # train_accuracy.reset_states()

        train_step(input_data)
        
        pred = model_dem.predict(input_data)
        loss_history.append({'i':epoch+1, 'loss':pred['total_energy'], 'id':0}) # 0 means adam, 1 means lbfgs

        if epoch % 10 == 0:
            print(f"Iter {epoch}: total_loss = {pred['total_energy']}, int = {pred['internal_energy']}, ext = {pred['external_energy']}")


    loss_history_to_csv(loss_history=loss_history, save_path=data_dir/'output/loss.csv')

    validation = model_x_to_result.predict(validation_data['X_val'])

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



    xs  = input_data['X_int'][0, :, 0]
    ys  = input_data['X_int'][0, :, 1]
    # val = model_data['X_int'][0, :, 0]

    viz_format.plot_mesh(xs, ys, val, contour_num=40, contour_line = False, cmap='turbo')
