""" Sample Ansys result

Sample ansys result from .txt and convert to csv at specified xs and ys.
And save original Ansys result in .csv format.
"""

from pathlib import Path
import numpy as np

import sys
sys.path.append("C:\\Users\\taku\\Hiroshima-U-Master\\OneDrive - Hiroshima University\\ドキュメント\\1kouza\\MasterResearch\\PythonCode\\ReadAnsysResult")
from sample_result import ResultSampler



#-------------------------------------------------------------------------------
# Script Begins
#-------------------------------------------------------------------------------
save_dir = Path('./')

# Sample position
xs = np.linspace(0.0, 5.0, num=50)
ys = np.full_like(xs, 0.0)

sampler = ResultSampler()
sampler.load_ansys_result(
    coord_file_path='./coord.txt',
    value_path_dict={
        'disp_x' : './disp_x.txt',
        'disp_y' : './disp_y.txt',
        'stress_x':'./stress_x.txt',
        'stress_y':'./stress_y.txt',
        })


sampler.save_original(save_dir/'Ansys.csv')
_ = sampler.sample_results(xs, ys, save_path='Ansys_sampled.csv')
