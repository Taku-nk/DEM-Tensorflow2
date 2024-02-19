import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.axes_grid1 import make_axes_locatable

class AnsysResult:
    def __init__(self, data_dir='', dx=0.02, unit='m'):
        self.data_dir = data_dir
        self.dx = dx

        # unit = 'm'
        self.key_x = f' X({unit})'
        self.key_y = f' Y({unit})'
        self.key_z = f' Z({unit})'

# Coordinate data frame
        col_slicer_coord = ('Node ID', self.key_x, self.key_y, self.key_z)
        coord_df = pd.read_csv(os.path.join(self.data_dir, "coord.txt"),
                                delimiter="\t")
        coord_df = coord_df.loc[:, col_slicer_coord]
        coord_df = coord_df.drop_duplicates(subset='Node ID')
        coord_df = coord_df.sort_values(by=['Node ID'])

        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')



# Disp data frame
        self.key_disp = f'Directional Deformation ({unit})'
        col_slicer_disp = ('Node Number', self.key_disp)
        disp_df = pd.read_csv(os.path.join(self.data_dir, "displacement.txt"), 
                                delimiter="\t")
        disp_df = disp_df.loc[:, col_slicer_disp]
        disp_df = disp_df.drop_duplicates(subset='Node Number')
        disp_df = disp_df.sort_values(by=['Node Number'])


        self.coord_df = coord_df
        self.disp_df = disp_df
        self.result_df = pd.DataFrame(
                {
                    'Node ID': self.coord_df.loc[:, 'Node ID'].to_numpy(),
                    self.key_x: self.coord_df.loc[:, self.key_x].to_numpy(),
                    self.key_y: self.coord_df.loc[:, self.key_y].to_numpy(),
                    self.key_z: self.coord_df.loc[:, self.key_z].to_numpy(),
                    self.key_z: self.coord_df.loc[:, self.key_z].to_numpy(),
                    self.key_disp: self.disp_df.loc[:, self.key_disp].to_numpy(),
                })


    def set_dx(self, dx):
        self.dx = dx

    
    def plot_result_3d(self, xs=None, ys=None, zs=None):
        """ Plot Exported Result """
        # init_xs = self.coord_df.loc[:, self.key_x].to_numpy()
        # init_ys = self.coord_df.loc[:, self.key_y].to_numpy()
        # disp_arr = self.disp_df.loc[:, self.key_disp].to_numpy()

        if xs is None:
            xs = self.result_df.loc[:, self.key_x].to_numpy() # initial x coord
        if ys is None:
            ys = self.result_df.loc[:, self.key_y].to_numpy()
        if zs is None:
            zs = self.result_df.loc[:, self.key_disp].to_numpy() # Displacement



        self.ax.set_title('displacement')
        self.ax.set_box_aspect((1,1,0.5))
# ax.set_zlim(-0.02, 0.02)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Displacement")

        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        # Plot color determined
        color_data = zs 

# c= float or 1D array(len=material point)
        sc = self.ax.scatter(xs, ys, zs,
                  # vmin=vmin, vmax=vmax, 
                  # vmin=0, vmax=1.5e+6, 
                   c=color_data, 
                   s=2,
                   cmap=plt.get_cmap('viridis_r'), depthshade=False)
        plt.colorbar(sc, shrink=0.8, pad=0.15)

        plt.show()


    def get_2d_image_data(self, x_start, x_end, y_start, y_end):
        """ 
            Get 2D iamge data for export. 
            Arguments:
                - x_start: double, coordinate of left edge (not index.) 
                - x_end  : double, coordinate of right edge (not index.)
                - y_start: double, coordinate of top edge (not index.)
                - y_start: double, coordinate of lower edge (not index.)

            Returns:
                - Tuple of 2D array: (xs, ys, zs), xs and ys are 
                initial coordinate, zs are result displacement.
        """
        # print(self.result_df)
        # coord_df = self.coord_df.round({self.key_x: 2, self.key_y:2})
        result_df = self.result_df.round({self.key_x:2, self.key_y:2})
        dx = self.dx

        
        # print(len(result_df.loc[(result_df[self.key_x]>=x_start)&
        #     (result_df[self.key_x]<=x_end)&
        #     (result_df[self.key_y]>=y_start)&
        #     (result_df[self.key_y]<=y_end)
        #     ]))

        # By using meshgrid, we can pickup nodes with the interval dx=0.02
        xs = np.linspace(x_start, x_end, int(round((x_end-x_start)/dx+1, 0))).round(2)
        ys = np.linspace(y_start, y_end, int(round((y_end-y_start)/dx+1, 0))).round(2)
        
        X, Y = np.meshgrid(xs, ys)
        grid_shape = X.shape
        X = np.reshape(X, (X.size, 1))
        Y = np.reshape(Y, (Y.size, 1))
        coord = np.hstack((X, Y))
        
        # Empty data frame 
        filtered_df = pd.DataFrame({
            'Node ID': pd.Series(dtype='int64'), 
            self.key_x :pd.Series(dtype='float64'), 
            self.key_y :pd.Series(dtype='float64'), 
            self.key_disp :pd.Series(dtype='float64'), 
            })
        
        # print(filtered_df)
        # i = 0
        for x, y in coord:
            # print(f"x = {x}, y = {y}")
            selected_node = result_df.loc[(result_df[self.key_x]==x) & (result_df[self.key_y]==y)] 
            if len(selected_node) > 1:
                print('Error: Duplicate node was found.')
                break
            elif len(selected_node) == 0:
                print(f'Error: Node at x={x}, y={y} is missing.')
                break
            else:
                # print(selected_node)
                filtered_df = pd.concat((filtered_df, selected_node)) 

        selected_X = filtered_df.loc[:, self.key_x].to_numpy()
        selected_Y = filtered_df.loc[:, self.key_y].to_numpy()
        selected_DISP = filtered_df.loc[:, self.key_disp].to_numpy()

        try:
            # this X_out will be a return value
            X_out = np.reshape(selected_X, grid_shape)
            Y_out = np.reshape(selected_Y, grid_shape)
            DISP_out = np.reshape(selected_DISP, grid_shape)
        except ValueError:
            print("ERROR : Some node data are missing in specified area.")

        # fig = plt.figure(figsize=(8,6))
        # ax0 = fig.add_subplot(211)
        # ax1 = fig.add_subplot(212)
        # ax2 = fig.add_subplot(221)

        # ax0.imshow(X_out)
        # ax1.imshow(Y_out)
        # ax2.imshow(DISP_out)
        # plt.show()
        # plt.imshow(X_out)
        return X_out, Y_out, DISP_out

        # print(grid_shape)
        # print()
            # if i >= 5:
            #     break
            # print(result_df.loc[(result_df[self.key_x]==x) & (result_df[self.key_y]==y)])
            # print(filtered_df)
            # i = i+1
            # input("Press Enter to continue...")
            # filtered_data.append()
        
    def export_result(self, file_name, edge_region, output_dir=None):
        """ Export result as numpy .npy
        Arguments:
            - file_name: str, file name to be saved, e.g. 'data.npy'
            - edge_region : tuple, (x_start, x_end, y_start, y_end)
            - output_dir: path like object, str, output directory.
                          If output_dir==None, result will be saved to 
                          Ansys result directory.
        """

        if output_dir==None:
            output_dir = self.data_dir

        file_path = os.path.join(output_dir, file_name)
        _, _, disp = self.get_2d_image_data(*edge_region)

        with open(file_path, 'wb')as f:
            np.save(f, disp)
            print(f"Result numpy file was saved to \'{file_path}\'")


        # print(coord)
        # print(ys)

        # print(coord_df)

        # print(self.coord_df.loc(self.coord_df[self.key_x]))

        # pass
        # init_xs = self.coord_df.loc[:, self.key_x].to_numpy()
        # init_ys = self.coord_df.loc[:, self.key_y].to_numpy()
        # disp_arr = self.disp_df.loc[:, self.key_disp].to_numpy()


# print(disp_df)

# for i in range(len(disp_df)):
    # print(disp_df.iloc[i, 0])
    # if disp_df.iloc[i, 0] != coord_df.iloc[i, 0]:
    #     print(f'Node ID at {i} miss-match')
    #     break
    # class ExportResult:
    #     """ Export AnsysResult """
    #     def __init__(self, ansys_result, ):
    # def Export_Ansys_Result(ansys_result, output_dir='', file_name='',)

        

if __name__ == '__main__':
    data_dir = "./result/"
    result = AnsysResult(data_dir)
    # result.plot_result_3d()
    # result_data = result.get_2d_image_data(0.51, 0.61, -0.61, 0.61)
    # result.plot_result_3d(*(result_data))
    # print(result_data[0].size)
    print(result.result_df)

    # 順番はすべてマイナス->プラスの順番.
    edge_xp = ( 0.51,  0.61, -0.61,  0.61)
    edge_xn = (-0.61, -0.51, -0.61,  0.61)
    edge_yp = (-0.61,  0.61,  0.51,  0.61)
    edge_yn = (-0.61,  0.61, -0.61, -0.51)
    

    result.export_result('edge_xp.npy', edge_xp)
    result.export_result('edge_xn.npy', edge_xn)
    result.export_result('edge_yp.npy', edge_yp)
    result.export_result('edge_yn.npy', edge_yn)
    # result.plot_result_3d(*(result.get_2d_image_data(*edge_yp)))
    
    # mask_regions = [edge_xp, edge_xn, edge_yp, edge_yn]
    # for edge in mask_regions:
    #     _, _, disp = result.get_2d_image_data(*edge)
    #     print(f"{edge=}")
        # disp.save()
    # size is varified
    # print(62*6 == result_data[0].size)




# print(coord_df.loc[:, key_x])

# print(init_ys.shape)





