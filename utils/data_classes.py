""" Prepare data for PINN.

This module contains useful tools like read and write data from or to .csv.
And visualize the csv data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class InputData:
    def __init__(
        self,
        internal_training_point: str,
        BC_Neumann_training_point: str,
        BC_Neumann_traction_vector: str,
        validation_point: str
    ) -> None:
        """ Retreave csv file and prepare.

        Retreave csv file created in the Blender.

        Args:
            csv directory for each data created in Blender.
        """
        self.internal_DF = pd.read_csv(internal_training_point)
        self.BC_Neumann_point_DF = pd.read_csv(BC_Neumann_training_point)
        self.BC_Neumann_load_DF = pd.read_csv(BC_Neumann_traction_vector)
        self.validation_point = pd.read_csv(validation_point)


    def get_data_for_Blender_DEM(self):
        """ Get DEM input data for Blender DEM
        This X_f contains 'E' for X_f[3]
        Args:
            None

        Returns:
            X_f: Array like, internal face point position and weight
                X_f[0]=x, X_f[1]=y, X_f[2]=area, X_f[3]='E'
            X_bnd: Array like, boundary edge point position and weight
                The order matters.
            Grid: Array like, x, y, weight.
                Grid[0]=x, Grid[1]=y, Grid[2]=area, Grid[3]='E'
        """

        X_f_DF = self.internal_DF.loc[:, ['xc', 'yc', 'area', 'E']]
        X_f = X_f_DF.to_numpy()

        X_b_coord_DF = self.BC_Neumann_point_DF.loc[:, ['xc', 'yc', 'length']]

        T_b_DF = self.BC_Neumann_load_DF.loc[:, ['Nx', 'Ny']]

        X_bnd_DF = pd.concat([X_b_coord_DF, T_b_DF], axis=1)
        X_bnd = X_bnd_DF.to_numpy()


        Grid_DF = self.validation_point.loc[:, ['xc', 'yc', 'area', 'E']]
        Grid = Grid_DF.to_numpy()

        return X_f, X_bnd,  Grid


    def get_data(self):
        """Get DEM input Numpy data.
        Args:
            None.
        Returns:
            input_data_dict: dict. Input data dictionary containing the following np.ndarray.
                'X_int'   : np.ndarray. Shape = (1, :, 2). Internal training point location x and y.
                'wt_int'  : np.ndarray. Shape = (1, :, 1). Internal training point weight (area).
                'X_bnd'   : np.ndarray. Shape = (1, :, 2). Neumann boundary training point location x and y.
                'wt_bnd'  : np.ndarray. Shape = (1, :, 1). Neumann boundary training point weight (length).
                'Trac_bnd': np.ndarray. Shape = (1, :, 2). Neumann boundary traction vector. (N/m^2)
                'X_val'   : np.ndarray. Shape = (1, :, 2). Internal validation point location x and
                'wt_val'  : np.ndarray. Shape = (1, :, 1). Internal validation point weight (area).

            Note.: The first '1' in the shape (1, :, ..) is the batch. 
            Second ':' is variable length data, that is the number of training points.
        """
        X_int = self.internal_DF.loc[:, ['xc', 'yc']].to_numpy()
        wt_int = self.internal_DF.loc[:, ['area']].to_numpy()

        X_bnd = self.BC_Neumann_point_DF.loc[:, ['xc', 'yc']].to_numpy()
        wt_bnd = self.BC_Neumann_point_DF.loc[:, ['length']].to_numpy()

        Trac_bnd = self.BC_Neumann_load_DF.loc[:, ['Nx', 'Ny']].to_numpy()

        X_val = self.validation_point.loc[:, ['xc', 'yc']].to_numpy()
        wt_val = self.validation_point.loc[:, ['area']].to_numpy()

        input_data_dict = {
                'X_int'   :    X_int[np.newaxis, :, :],
                'wt_int'  :   wt_int[np.newaxis, :, :],
                'X_bnd'   :    X_bnd[np.newaxis, :, :],
                'wt_bnd'  :   wt_bnd[np.newaxis, :, :],
                'Trac_bnd': Trac_bnd[np.newaxis, :, :],
                'X_val'   :    X_val[np.newaxis, :, :],
                'wt_val'  :   wt_val[np.newaxis, :, :],
            }

        return input_data_dict

