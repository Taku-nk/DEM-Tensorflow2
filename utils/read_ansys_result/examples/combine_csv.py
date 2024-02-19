""" Combine .csv saved by ResultSampler.sample_result.

if you wanna save all value at once you can use ResultSampler.sample_results().
But if you fogot to use it and used ResultSampler.sample_result() one by one, 
still you can combine results.csv by using combine_csv_results function.
"""
import sys
sys.path.append("C:\\Users\\taku\\Hiroshima-U-Master\\OneDrive - Hiroshima University\\ドキュメント\\1kouza\\MasterResearch\\PythonCode\\ReadAnsysResult")
from sample_result import combine_csv_results

combine_csv_results(
    ['./sampled_stress_x.csv',
     './sampled_stress_y.csv',],
    save_path='./DEM_results.csv')

