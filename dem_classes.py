import tensorflow as tf
from loss_utils import loss_history_to_csv
import numpy as np


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
        """X to displacement and stress, energy density.
        Args:
            X: shape(:, :, 2), X[:, :, 0] is x and X[:, :, 1] is y.
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
    



class AnalysisDEM:
    """This is the top most DEM analysis class
    """
    def __init__(self, layer_x_to_disp, E=1.0, nu=0.3):
        self.model_x_to_result = ModelXToResult(layer_x_to_disp, E=E, nu=nu)
        self.model_dem = ModelDEM(self.model_x_to_result)
        self._build_dem()
    
        self.loss_obj = LossDEM()
        self.optimizer = tf.keras.optimizers.Adam()

        self.loss_history = []


    def train(self, input_data, epochs=1):
        """Train the model_dem
        """
        self.model_dem.compile(
        optimizer=self.optimizer,
        loss=self.loss_obj
        )

        for epoch in range(epochs):
            # Reset the metrics at the start of the next epoch
            # train_loss.reset_states()
            # train_accuracy.reset_states()

            self._train_step(input_data)
            
            pred_energy = self.model_dem.predict(input_data)
            self.loss_history.append({'i':epoch+1, 'loss':pred_energy['total_energy'], 'id':0}) # 0 means adam, 1 means lbfgs

            if epoch % 10 == 0:
                print(f"Iter {epoch}: total_loss = {pred_energy['total_energy']}, int = {pred_energy['internal_energy']}, ext = {pred_energy['external_energy']}")
                
        return
    

    def save_history(self, save_path):
        """Save result
        Args:
            save_dir: Pathlib path. In this path the the result will be saved.
        """
        loss_history_to_csv(loss_history=self.loss_history, save_path=save_path)

        return
    

    def predict(self, X):
        """Predict the result.
        Args:
            X: np.ndarray. Shape=(1, :, 2)
        Returns:
            result_dict: dict. The result contains the following.
            {   
                'disp_x': shape=(1, :, 1),
                'disp_y': shape=(1, :, 1),
                'stress_x': shape=(1, :, 1),
                'stress_y': shape=(1, :, 1),
                'stress_xy': shape=(1, :, 1),
                'strain_energy_density': shape=(1, :, 1)
            }
        """
        return self.model_x_to_result.predict(X)





    def _build_dem(self):
        self.model_dem({
        'X_int'    : tf.keras.Input(shape=(None, 2)),
        'wt_int'   : tf.keras.Input(shape=(None, 1)),
        'X_bnd'    : tf.keras.Input(shape=(None, 2)),
        'wt_bnd'   : tf.keras.Input(shape=(None, 1)),
        'Trac_bnd' : tf.keras.Input(shape=(None, 2)),
        })
        self.model_dem.summary()
        return
  

    @tf.function
    def _train_step(self, input_data):
        dummy_label = np.array(0.0, dtype=np.float32)
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            pred_energy = self.model_dem(input_data, training=True)
            loss = self.loss_obj(dummy_label, pred_energy['total_energy'])
        gradients = tape.gradient(loss, self.model_dem.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model_dem.trainable_variables))
        return
