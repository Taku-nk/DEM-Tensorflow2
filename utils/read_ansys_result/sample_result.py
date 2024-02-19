"""Samples 2D analysis result

With this class, you can sample linearly interpolated where ever you want.
And you can save the original and sampled data.
In Ansys, you have to export whole value and selection info can be smaller 
regeon than value. Value txt file has inlcude the regeon of selection info.

Typical usage:

With Numpy result:
    sampler = ResultSampler()
    sampler.load_numpy_result(xs, ys,
        {'key1':values1, 'key2':values2, ...}) # generate pandas dataframe

    print(sampler.get_result_df())
    sampled_result = sampler.sample_result(xs, ys, 'key1', save_path='./result.csv') # linear interpolate and sample


With .csv Ansys result:
    sampler = ResultSampler()
    sampler.load_ansys_result(
        coordinate_path= './coord.txt',
        {
            'disp_x':'./displacement_x.txt',
            'disp_y':'./displacement_y.txt',
            'stress_x':'./ stress_x.txt',...
        }
    )

    print(sampler.get_result_df())
    sampled_result = sampler.sample_result(xs, ys, 'key1', save_path) # linear interpolate and sample
"""

import pandas as pd
import numpy as np
from scipy.interpolate import griddata

class ResultSampler:
    """ Sample analysis result and visualize 
    
    Currently only varified to work on 2D analys.
    Basically this class read result into Pandas DataFrame as an intermediate
    data. Keys of DataFrame will be 'Node ID','x', 'y', 'user_def1', 'user_def2', ...
    """

    def __init__(self):
        self.result_df = pd.DataFrame()

    
    def load_ansys_result(self, coord_file_path, value_path_dict ,is_3d=False):
        """ Load Ansys result txt.

        Load csv and create pandas data frame. First, load selected coordinate
        (the Ansys 'selection information'). Next, read values e.g displacement,
        and append values to self.result_df, only where selected, which means
        only corresponding value to the coordinate.

        Args:
            coord_file_path: str. Path to the txt result file.
            value_path_dict: dictionary containing 'key' and corresponding file
            path string. Example:
                    {'disp_x':'./result/displacement_x',
                     'disp_y':'./result/displacement_y'}
        Returns:
            None
        """
        self.result_df = pd.DataFrame()

        coord_df = pd.read_csv(coord_file_path, delimiter="\t")
        coord_df = coord_df.iloc[:, [4, 1, 2]] # 4:Node ID, 1:x, 2:y
        coord_df = coord_df.drop_duplicates(subset='Node ID')
        coord_df = coord_df.sort_values(by=['Node ID'])
        coord_df = coord_df.set_axis(['Node ID', 'x', 'y'], axis='columns')
        coord_df.set_index(keys='Node ID', drop=False, inplace=True)

        self.result_df = coord_df


        # Node Number == Node ID in Ansys
        for key_i, path_i in value_path_dict.items():
            df = pd.read_csv(path_i, delimiter="\t")
            df = df.drop_duplicates(subset='Node Number')
            df.set_index(keys='Node Number', inplace=True) # column as index

            # Save only selected data in coord df (in Ansys, selection info)
            selected_id_mask = coord_df.loc[:, 'Node ID']
            df = df.loc[selected_id_mask, :] # mask with pands series
            df = df.sort_values(by=['Node Number'])
            df = df.set_axis([key_i], axis='columns')
            
            self.result_df[key_i] = df # append data


        self.result_df.drop(columns=['Node ID'], inplace=True)


        # print(self.result_df.head())
    def load_numpy_result(self, xs, ys, value_dict, is_3d=False):
        """ Load numpy result
        Load numpy and create pandas data frame for later use, and save it to
        self.result_df.

        Args:
            xs: array like. shape (n,). Original position of x
            ys: array like. shape (n,). Original position of y
            value_path_dict: dictionary containing 'key' and corresponding numpy
            Example:
                    {'disp_x': u_pred_np,
                     'disp_y': v_pred_np}
        Returns:
            None
        """
        self.result_df = pd.DataFrame()
        coord_df = pd.DataFrame({'x':xs, 'y':ys})
        values_df = pd.DataFrame(value_dict)

        self.result_df = pd.concat((coord_df, values_df), axis=1)
        # print(self.result_df)


    def load_csv_result(self, file_path):
        """load from saved csv result"""
        self.result_df = pd.read_csv(file_path)



    def get_result_df(self):
        """Getter of result data frame"""
        return self.result_df
            

    def sample_result(self, xs, ys, key, save_path=''):
        """Sample interpolated result from original data.
        
        This method uses scipy.interpolate.griddata function for interpolation.
        griddata function returns nan value if sampled point is out of bounds.
        This sampling can be 2D area or 1D line. Just specfy where you want to
        sample by (xs, ys).

        Args:
            xs:  Array like. 1D array. shape=(n, ).
            ys:  Array like. 1D array. shape=(n, ).
            key: str, key name specified while loading data.
            save_path: str, if you want to save sampled result, then give path.
                if you specify nothing, then no saving happen.

        Returns:
            values: Array like. 1D array. shape=(n, ).
        """
        orig_xs = self.result_df.loc[:, 'x']
        orig_ys = self.result_df.loc[:, 'y']
        orig_values = self.result_df.loc[:, key]


        interpolated_value = griddata((orig_xs, orig_ys),
                                      orig_values,
                                      (xs, ys), # sample position
                                      method='linear')
        
        
        if save_path != '':
            self.__save_sampled_result(xs, ys, interpolated_value, key, save_path)

        return interpolated_value


    def sample_results(self, xs, ys, save_path=''):
        """Sample interpolated result from original data. 
        
        This method uses scipy.interpolate.griddata function for interpolation.
        griddata function returns nan value if sampled point is out of bounds.
        This sampling can be 2D area or 1D line. Just specfy where you want to
        sample by (xs, ys).

        Args:
            xs:  Array like. 1D array. shape=(n, ).
            ys:  Array like. 1D array. shape=(n, ).
            save_path: str, if you want to save sampled result, then give path.
                if you specify nothing, then no saving happen.

        Returns:
            values: Pandas DataFrame. shape=(n, len(keys)+xs+ys).
        """
        sampled_df = pd.DataFrame({'x':xs, 'y':ys}) 
        value_keys = self.result_df.columns[2:] # exclude 'x' 'y'
        
        for key in value_keys:
            sampled_value = self.sample_result(xs, ys, key)
            sampled_df[key] = sampled_value


        if save_path != '':
            sampled_df.to_csv(save_path, index=False)
            print("Sampled data was saved to '{}'".format(save_path))
        

        return sampled_df 
        # return sampled_df.head();
            
            



    
    def __save_sampled_result(self, xs, ys, values, key, save_path):
        """Save sampled result using pandas DataFrame.
        """
        df = pd.DataFrame({'x':xs, 'y':ys, key:values})
        # print(df)
        df.to_csv(save_path, index=False)
        print("Sampled data was saved to '{}'".format(save_path))
     

    def save_original(self, file_path):
        """ Save original data frame """
        self.result_df.to_csv(file_path, index=False)
        print("Original data was save to '{}'".format(file_path))
    

def combine_csv_results(paths, save_path=''):
    """ Combine separetely saved csv into one

    Args:
        paths: list of csv file path to be combined. csv file header must be
            'x', 'y', 'value'

    Returns:
        None
    """
    combined_df = pd.read_csv(paths[0]).loc[:, ['x', 'y']]


    for path in paths:
        value_df = pd.read_csv(path).iloc[:, -1] # first two are 'x','y' last one is 'value'
        combined_df = pd.concat((combined_df, value_df), axis=1)

    # print(combined_df.head())

    if save_path != '':
        combined_df.to_csv(save_path, index=False)
        print("Combined data was saved to '{}'".format(save_path))




        
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    sampler = ResultSampler()
    sampler.load_ansys_result(
        coord_file_path='./result_whole/coord.txt',
        value_path_dict={
            'disp_x' : './result_whole/disp_x.txt',
            'disp_y' : './result_whole/disp_y.txt',
            'stress_x':'./result_whole/stress_x.txt',
            'stress_y':'./result_whole/stress_y.txt',
            }
    )

    # sampler.save_original('./result_whole/result_Ansys.csv')


    df = sampler.get_result_df()
    
    # print(df)

    # xs = df.loc[:, 'x'].to_numpy()
    # ys = df.loc[:, 'y'].to_numpy()
    # u_pred = df.loc[:, 'disp_x'].to_numpy()
    # v_pred = df.loc[:, 'disp_y'].to_numpy()
    # sigma_y_pred = df.loc[:, 'stress_y'].to_numpy()

    # sampler_np = ResultSampler()
    # sampler_np.load_numpy_result(
    #     xs, ys, 
    #     value_dict={
    #         'disp_x':u_pred,
    #         'disp_y':v_pred,
    #         'stress_y':sigma_y_pred
    #         })

    
    # x_grid = np.linspace(1.5, 5, num=50)
    # y_grid = np.linspace(-1, 1, num=20)

    # X,Y = np.meshgrid(x_grid, y_grid)
    # xs = X.flatten()
    # ys = Y.flatten()

    xs = np.linspace(1.5, 5, num=50)
    ys = np.zeros_like(xs)
    sampler.sample_results(xs, ys, save_path='./result_whole/ansys_dropped_ID.csv')

    # values = sampler.sample_result(xs, ys, 'stress_y', 
    #             save_path='./result_whole/sampled_stress_y_ansys.csv')

    # _ = sampler.sample_result(xs, ys, 'stress_x', 
    #             save_path='./result_whole/sampled_stress_x_ansys.csv')

    # b = plt.scatter(xs, ys, c=values)
    # saved_df = pd.read_csv('./result_whole/sampled_stress_y_ansys.csv')
    combine_csv_results(
            [
                './result_whole/sampled_stress_x_ansys.csv',
                './result_whole/sampled_stress_y_ansys.csv',
            ],
            save_path = './result_whole/combined_values.csv')
    # print(saved_df.head())

    # print(sampler.sample_results(xs, ys, save_path='./result_whole/sampled_all.csv'))
    


    # fig = plt.figure(figsize=(6,4))
    # ax = fig.add_subplot()

    # ax.axis('equal')
    # ax.grid(ls=':')

    # b=ax.scatter(xs, ys, c=values)
    # plt.colorbar(b)
    
    # plt.plot(xs, values)
    # ax.plot(xs, values)
    # plt.show()
    # print(sampler.get_result_df())








