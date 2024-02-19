import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ReadAnsysResult:
    def __init__(self, coord_path='', disp_path='',  unit='m'):
        ''' return Pandas DataFrame along center x '''


        # unit = 'm'
        self.key_x = f' X({unit})'
        self.key_y = f' Y({unit})'
        self.key_z = f' Z({unit})'

        self.key_disp = f'Directional Deformation ({unit})'

        # Coordinate data frame
        col_slicer_coord = ('Node ID', self.key_x, self.key_y, self.key_z)
        coord_df = pd.read_csv(coord_path, delimiter="\t")
        coord_df = coord_df.loc[:, col_slicer_coord]
        coord_df = coord_df.drop_duplicates(subset='Node ID')

        # coord_df = coord_df.sort_values(by=['Node ID'])
        coord_df = coord_df.sort_values(by=[self.key_x])

        # disp_df = pd.read_csv(disp_path, delimiter="\t", index_col='Node Number') 
        disp_df = pd.read_csv(disp_path, delimiter="\t") 
        disp_df = disp_df.set_index('Node Number', drop=False)
        disp_df = disp_df.drop_duplicates(subset='Node Number')

        node_ID_mask = coord_df['Node ID']
        disp_df = disp_df.loc[node_ID_mask]
        
        # Combine coordinate dataframe and disp_df
        self.result_df = pd.DataFrame(
                {
                    'Node ID':  coord_df.loc[:, 'Node ID'].to_numpy(),
                    self.key_x: coord_df.loc[:, self.key_x].to_numpy(),
                    self.key_y: coord_df.loc[:, self.key_y].to_numpy(),
                    self.key_z: coord_df.loc[:, self.key_z].to_numpy(),
                    self.key_z: coord_df.loc[:, self.key_z].to_numpy(),
                    self.key_disp: disp_df.loc[:, self.key_disp].to_numpy(),
                })

        # print()
    def get_dataframe(self):
        """ Returns result dataframe """
        return self.result_df

    def save_result_df(self, output_dir=''):
        """ Save result dataframe """
        output_file = os.path.join(output_dir, 'result.csv')
        self.result_df.to_csv(output_file)
        print("Result dataframe was saved to \'{}\'".format(output_file)) 






if __name__ == '__main__':
    output_dir = './result'
    
    coord_path = './result/coord_center_x.txt'
    disp_path = './result/disp_center_x.txt'
    
    result = ReadAnsysResult(coord_path, disp_path)
    # result.save_result_df(output_dir)
    
    result_df = result.get_dataframe()
    print(result_df)


    xs = result_df.loc[:, ' X(m)'].to_numpy()
    disp = result_df.loc[:, 'Directional Deformation (m)'].to_numpy()

    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot()
    ax.plot(xs, disp)
    plt.show()



