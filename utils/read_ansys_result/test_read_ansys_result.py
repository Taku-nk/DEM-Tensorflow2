import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.axes_grid1 import make_axes_locatable

unit = 'm'
key_x = f' X({unit})'
key_y = f' Y({unit})'
key_z = f' Z({unit})'

# Coordinate data frame
col_slicer_coord = ('Node ID', key_x, key_y, key_z)
coord_df = pd.read_csv("./result/coord.txt", delimiter="\t")
coord_df = coord_df.loc[:, col_slicer_coord]
coord_df = coord_df.drop_duplicates(subset='Node ID')
coord_df = coord_df.sort_values(by=['Node ID'])
# print(coord_df)
# print(len(coord_df.drop_duplicates(subset='Node ID')))

# Disp data frame
key_disp = f'Directional Deformation ({unit})'
col_slicer_disp = ('Node Number', key_disp)
disp_df = pd.read_csv("./result/displacement.txt", delimiter="\t")
disp_df = disp_df.loc[:, col_slicer_disp]
disp_df = disp_df.drop_duplicates(subset='Node Number')
disp_df = disp_df.sort_values(by=['Node Number'])
# print(disp_df)

# for i in range(len(disp_df)):
    # print(disp_df.iloc[i, 0])
    # if disp_df.iloc[i, 0] != coord_df.iloc[i, 0]:
    #     print(f'Node ID at {i} miss-match')
    #     break


# print(coord_df.loc[:, key_x])
init_xs = coord_df.loc[:, key_x].to_numpy()
init_ys = coord_df.loc[:, key_y].to_numpy()
disp_arr = disp_df.loc[:, key_disp].to_numpy()

# print(init_ys.shape)





fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(221, projection='3d')

ax.set_title('displacement')
ax.set_box_aspect((1,1,0.5))
# ax.set_zlim(-0.02, 0.02)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Displacement")

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False


color_data = disp_arr 

# c= float or 1D array(len=material point)
sc = ax.scatter(init_xs, init_ys, disp_arr,
          # vmin=vmin, vmax=vmax, 
          # vmin=0, vmax=1.5e+6, 
           c=color_data, 
           s=2,
           cmap=plt.get_cmap('viridis_r'), depthshade=False)
plt.colorbar(sc, shrink=0.8, pad=0.15)

plt.show()
