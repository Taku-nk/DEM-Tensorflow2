# Deep Energy Method
* Author: Taku Nakagawa
* Organization: Hiroshima University

## Prerequisite
This program needs working TensorFlow 2.8. Following is the required dependencies.
Detailed compatibility can be found on https://www.tensorflow.org/install/source#gpu .

* Python 3.10
* Nvidia driver
* CUDA toolkit 11.2
* cuDNN 8.1.1

Also needs following Python modules.
* TensorFlow 2.8.0
* Numpy 1.25.2
* Scipy 1.11.2
* Pandas
* Matplotlib

## How to use
1. Prepare input .csv file in "analysis_data"
2. Run `python dem_analysis_crack.py` for center crack analysis. Or run `python dem_analysis.py` for simple tensile analysis without crack.
3. You can found the result under "analysis_data"

Note.
- If you want to use custom model, then refer to 'PythonScript'.
- If you want to visualize the result, then refer to 'Post_DEM_Example'
