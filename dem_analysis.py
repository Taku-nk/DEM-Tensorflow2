
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from loss_utils import loss_history_to_csv
sys.path.append("/home/nakagawa/mylib/PythonCode/visualize_2D")
from visualize_DEM_2D import Visualize2DFormat as VizFormat

sys.path.append(
    "/home/nakagawa/mylib/PythonCode/ReadAnsysResult")
from sample_result import ResultSampler


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



class LayerXToDispGrad(tf.keras.layers.Layer):
    def __init__(self, layer_x_to_disp):
        """Calculate displacement and gradient of displacement.
        """
        super(LayerXToDispGrad, self).__init__()
        self.layer_x_to_disp = layer_x_to_disp

    def call(self, X):
        x, y = X[:, :, 0:1], X[:, :, 1:2]
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            g.watch(y)
            u, v= self.layer_x_to_disp(tf.concat([x, y], axis=-1)) # TODO: needs to know x and y so you have to again concat. It is not good 
        u_x = g.gradient(u, x)
        u_y = g.gradient(u, y)

        v_x = g.gradient(v, x)
        v_y = g.gradient(v, y)
        del g

        grad_u = tf.concat([u_x, u_y], axis=-1)
        grad_v = tf.concat([v_x, v_y], axis=-1)

        return u, v, grad_u, grad_v



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


    

class ModelXToResult(tf.keras.Model):
    def __init__(self, layer_x_to_disp, E=1.0, nu=0.3):
        """Model of X to loss and everything.
        """
        super(ModelXToResult, self).__init__()
        self.layer_disp_grad = LayerXToDispGrad(layer_x_to_disp)

        self.E = E
        self.nu = nu
        self.c11 = self.E/(1-self.nu**2)
        self.c22 = self.E/(1-self.nu**2)
        self.c12 = self.E*self.nu/(1-self.nu**2)
        self.c21 = self.E*self.nu/(1-self.nu**2)
        self.c31 = 0.0
        self.c32 = 0.0
        self.c13 = 0.0
        self.c23 = 0.0
        self.c33 = self.E/(2*(1+self.nu))


    def call(self, X):
        """X to Result.
        Args:
            X
        Returns:
            result_dictionary
            'disp_x':  shape(:, :, 1)
            'disp_y':  shape(:, :, 1)
            ...
        """
        u, v, grad_u, grad_v = self.layer_disp_grad(X)
        u_x, u_y = grad_u[:, :, 0:1], grad_u[:, :, 1:2]
        v_x, v_y = grad_v[:, :, 0:1], grad_v[:, :, 1:2]
        u_xy = u_y + v_x

        stress_x = self.c11*u_x + self.c12*v_y
        stress_y = self.c21*u_x + self.c22*v_y
        stress_xy = self.c33*u_xy
        
        strain_energy_density = 0.5*(stress_x*u_x + stress_y*v_y + stress_xy*u_xy)




        return {
            'disp_x':u,
            'disp_y':v,
            'stress_x':stress_x,
            'stress_y':stress_y, 
            'stress_xy':stress_xy, 
            'strain_energy_density':strain_energy_density
        } 


class ModelDEM(tf.keras.Model):
    def __init__(self, model_x_to_result):
        super(ModelDEM, self).__init__()
        self.model_x_to_result = model_x_to_result

    
    # def call(self, X_int, wt_int, X_bnd, wt_bnd, Trac_bnd):
    def call(self, inputs):
        X_int    = inputs['X_int']
        wt_int   = inputs['wt_int']
        X_bnd    = inputs['X_bnd']
        wt_bnd   = inputs['wt_bnd']
        Trac_bnd = inputs['Trac_bnd']



        result_int = self.model_x_to_result(X_int)
        result_bnd = self.model_x_to_result(X_bnd)

        trac_x = Trac_bnd[:, :, 0:1] # traction vector x component
        trac_y = Trac_bnd[:, :, 1:2] # traction vector y component


        int_energy = tf.reduce_sum(result_int['strain_energy_density'] * wt_int) # scalar no batch.
        ext_energy = tf.reduce_sum((result_bnd['disp_x'] * trac_x + result_bnd['disp_y'] * trac_y) * wt_bnd)
        total_energy = int_energy - ext_energy


        return {
            'internal_energy': int_energy,
            'external_energy': ext_energy,
            'total_energy': total_energy,
        }


class LossDEM(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return y_pred


 


if __name__=='__main__':
    data_dir = Path('./data')

    input_data = InputData(
        internal_training_point    = data_dir/"input/internal_point_and_weight.csv",
        BC_Neumann_training_point  = data_dir/"input/Neumann_BC_point_and_weight.csv",
        BC_Neumann_traction_vector = data_dir/"input/Neumann_BC_traction_vector.csv",
        validation_point           = data_dir/"input/validation_point_and_weight.csv"
    )

    input_data, validation_data = input_data.get_data() # dictionary basic shape (1, :, 1) or (1, :, 2)

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
    # model_dem(
    #     X_int = tf.keras.Input(shape=(None, 2)),
    #     wt_int = tf.keras.Input(shape=(None, 1)),
    #     X_bnd = tf.keras.Input(shape=(None, 2)),
    #     wt_bnd = tf.keras.Input(shape=(None, 1)),
    #     Trac_bnd = tf.keras.Input(shape=(None, 2)),
    # )    
    model_dem.summary()
    pred_energy = model_dem(input_data)
    # pred_energy = model_dem(
    #     X_int    = input_data['X_int'],
    #     wt_int   = input_data['wt_int'],
    #     X_bnd    = input_data['X_bnd'],
    #     wt_bnd   = input_data['wt_bnd'],
    #     Trac_bnd = input_data['Trac_bnd']
    # )
    # print(pred_energy['internal_energy'])
    # print(pred_energy['external_energy'])
    # print(pred_energy['total_energy'])
    total_energy = pred_energy['total_energy']
    loss_obj = LossDEM()
    dummy_label = np.array(0.0, dtype=np.float32)

    optimizer = tf.keras.optimizers.Adam()
    

    model_dem.compile(
        optimizer=optimizer,
        loss=loss_obj
    )
    # model_dem.fit(input_data, dummy_label)

    @tf.function
    def train_step(input_data):
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
