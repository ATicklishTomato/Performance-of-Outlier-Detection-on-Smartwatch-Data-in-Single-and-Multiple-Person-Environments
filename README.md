# Performance-of-Outlier-Detection-on-Smartwatch-Data-in-Single-and-Multiple-Person-Environments
### An analysis of the performance of different outlier detection methods on consumer-grade wearable data in environments with single and multiple subjects

Luuk Wubben

Bachelor Thesis as part of the 2023 Q4 instance of the [Research Project](https://github.com/TU-Delft-CSE/Research-Project) at [TU Delft](https://https//github.com/TU-Delft-CSE)

Date of the thesis award: 2023-06-30

## About the Project
In this repository are the Jupyter Notebook and scripts used to implement, verify, and test an implementation of a Gaussian Mixture Model (GMM) and Density-Based Spacial Clustering of Applications with Noise (DBSCAN) for the purpose of outlier detection. Both were implemented using scikit-learn's classes for the respective algorithms.

### The Data
The data used for this was timeseries data with heart rate and step count measurements. If you desire to run the code, you must provide your own data. The data used for this project was provided by the ME-TIME study, hosted at ClinicalTrials.gov with unique ID NCT05802563. This data could not be published with the paper or the code. Data should thus have the following format, in a Pandas DataFrame:
| time  | hr  | steps  |
|---|---|---|
| datetime  | numeric  | numeric  |

### Running the code
It is important to first install the necessary libraries for the code. The code requires:
|Name|Website|
|---|---|
|NumPy|https://numpy.org/|
|pandas|https://pandas.pydata.org/|
|scikit-learn|https://scikit-learn.org/stable/|
|matplotlib|https://matplotlib.org/|
|scipy|https://scipy.org/|
|librosa|https://librosa.org/|
|tqdm|https://tqdm.github.io/|
|kneebow|https://pypi.org/project/kneebow/|

With these installed, either in the Jupyter Notebook environment or the environment for the scripts, the code can be run. Within the Jupyter Notebook, there is a main setup block at the top of the file where parameters can be set. Every algorithm has a secondary parameter block for the single and multiple person environment setup to allow for specific parameter switches per algorithm and environment.

The script has base parameters set to allow for runs as they were done to collect data for the paper. These can be changed either by locally editing the code, or by passing new arguments to the function when calling it.

## License
Copyright © 2023 Luuk Wubben

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
