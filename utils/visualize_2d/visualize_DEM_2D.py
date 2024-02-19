
"""
    import sys
    sys.path.append("C:\\Users\\taku\\Hiroshima-U-Master\\OneDrive - Hiroshima University\\ドキュメント\\1kouza\\MasterResearch\\PythonCode\\ReadExcel2D")
 
    from VizExcel import ExcelViewer, PlotExcel

    file_path = '../results.xlsx'

    excel_result = ExcelViewer(file_path)
    print(excel_result.GetDataFRame())

    plotter = PlotExcel()
    #plotter.plotResult(excel_result, key='S11', hole_radius=1.0)
    plotter.plotResult(excel_result, key='S11', hole_radius=None)
"""
# 2D Excel contour Plotter with arbitrary xyz value.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import ticker

from mpl_toolkits.axes_grid1 import make_axes_locatable


class ExcelViewer():
    def __init__(self, file_path):
        """ Open excel file and view 
        Arguments:
            - file_path: str, file_path
        """
        self.file_path = file_path
        self.resultDataFrame = pd.read_excel(file_path)




     
    def GetDataFRame(self):
        """ Returns DataFrame """
        return self.resultDataFrame






class PlotExcel():
    def __init__(self):
        # self.file_dir = file_dir
        self.fig = plt.figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)


    def plotResult(self, excelViewer, key='S11', hole_radius=None, contour_num=20):
        """ 
        Arguments
        - excel Viewer: ExcelViewer object,
        - key: str, header name for coloring the plot.
        - hole_radius: float, this is applyed only for center hole models. 
                       Set None for non hole object.
        """
        result_DataFrame = excelViewer.GetDataFRame()

        x = result_DataFrame.loc[:, 'x'].to_numpy()
        y = result_DataFrame.loc[:, 'y'].to_numpy()
        value = result_DataFrame.loc[:, key].to_numpy()

        # triangles to render for surface plot
        triang = tri.Triangulation(x, y)

        # If you did not set hole_radius, then no masking happens.
        if hole_radius is not None:
            tri_center_x = x[triang.triangles].mean(axis=1)
            tri_center_y = y[triang.triangles].mean(axis=1)
            mask = np.where(((tri_center_x**2+tri_center_y**2)<hole_radius**2), 1, 0)
            triang.set_mask(mask)
        

        # self.ax.tricontourf(x,y,value, 20)
        contour = self.ax.tricontourf(triang, value, contour_num)
        self.ax.tricontour(triang, value, contour_num, colors='w', linewidths=0.5, linestyles='solid')
        self.ax.set_aspect('equal')
        self.ax.set_title(f'{key}')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

        plt.colorbar(contour, shrink=1, pad=0.05)
        plt.tight_layout()
        plt.show()



def debug_visualize_mesh(x, y, value, contour_num=20, contour_line=True):
    """ This function visualize data using triang.

    Args: 
        x    : array like. 
        y    : array like.
        value: array like.
        contour_num: int. The number of contour line to be drawn.

    Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    
    triang = tri.Triangulation(x, y)

    

    contour = ax.tricontourf(triang, value, contour_num)
    
    if contour_line:
        ax.tricontour(triang, value, contour_num, colors='w', linewidths=0.5, linestyles='solid')

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.colorbar(contour, shrink=1, pad=0.05)
    plt.tight_layout()
    plt.show()
 


class Visualize2DFormat:
    def __init__(self, figsize=(4, 3), fontsize=8):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)

        self.fontsize = fontsize 
        self.small_fac = 0.8

        self.config()


    def config(self):
        plt.rcParams["savefig.directory"] = "/Users/taku/Desktop"
        # plt.rcParams["savefig.format"] = 'eps'
        plt.rcParams["savefig.format"] = 'svg'
        # plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['mathtext.fontset'] = 'cm'
        self.ax.locator_params(axis='x',nbins=6)
        self.ax.locator_params(axis='y',nbins=6)
        self.ax.tick_params(axis='both', labelsize=self.fontsize*self.small_fac)

        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        # formatter.set_powerlimits((-1,1))
        self.ax.yaxis.set_major_formatter(formatter)

        self.ax.yaxis.offsetText.set_fontsize(self.fontsize)

        # self.ax.set_xlim(-5, 5)

        # self.ax.set_title("Displacement z", fontsize=16, fontname="Times New Roman")
        self.ax.set_xlabel("Coordinate $x$ [m]", fontsize=self.fontsize)
        self.ax.set_ylabel("Value $u$ [m]", fontsize=self.fontsize)
        # self.ax.grid(ls=':')

        self.ax.set_aspect('equal')


    def plot_mesh(self, x, y, value, contour_num=20, contour_line=True, cmap='viridis', save=True, fname='', extend='neither'):
        """ This function visualize data using triang.

        Args: 
            x    : array like. 
            y    : array like.
            value: array like.
            contour_num: int. The number of contour line to be drawn.
            contour_line: bool. Specify whether plot contour line or not.

        Returns:
            None
        """
        
        triang = tri.Triangulation(x, y)

        contour = self.ax.tricontourf(triang, value, contour_num, cmap=cmap, extend=extend)

        if contour_line:
            self.ax.tricontour(triang, value, contour_num, colors='w', linewidths=0.5, linestyles='solid')


        divider = make_axes_locatable(self.ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05) 
        cax = divider.append_axes("right", size=0.15, pad=0.05) 

        # plt.colorbar(contour, shrink=1, pad=0.05)
        cbar = plt.colorbar(contour, cax=cax)
        cbar.ax.tick_params(labelsize=self.fontsize*self.small_fac)

        plt.tight_layout()
        if save==True:
            plt.savefig(fname, bbox_inches='tight')

        plt.show()


    # def show(self):
    #     self.ax.legend(fontsize=self.fontsize*self.small_fac)
    #     plt.tight_layout()
    #     plt.show()
    


if __name__ == '__main__':
    # file_path = './exact.xlsx'
    # file_path = './results.xlsx'
    file_path = '../results.xlsx'

    excel_result = ExcelViewer(file_path)
    # print(excel_result.GetDataFRame())

    plotter = PlotExcel()
    plotter.plotResult(excel_result, key='S11', hole_radius=1.0)
    # plotter.plotResult(excel_result, key='S11', hole_radius=None)

