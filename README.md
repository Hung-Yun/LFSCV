# Elastic-Net Regression Based Model

> Reference: Kishida 2016 PNAS: *Subsecond dopamine fluctuations in human striatum
encode superposed error signals about actual and
counterfactual reward*.

### Package dependency:
- xlwings
- pickle
- glob

## Brief introduction

"We sought to develop a novel approach that uses in vitro calibration data to fit a cross-validated penalized linear regression model (elastic net (EN)-based approach) for estimating dopamine concentrations from **non–background-subtracted** voltammograms."

## Difference between Kishida's method and ours

Kishida used MATLAB's cvglmnet package, but we're using python's sklearn models.
The main two parameters we are looking for is alpha and lambda in his paper.
However, the nomenclatures in MATLAB are different from those in python, which is compared below.

| Kishida           	| Ours             	|
|-------------------	|------------------	|
| MATLAB (cvglmnet) 	| Python (sklearn) 	|
| alpha             	| l1_ratio         	|
| lambda            	| alpha            	|

### Method for optimizing model parameters

"The best α is determined via grid search (α for a range of λ);
we searched α values between from 0 to 1 in 0.1 increments.
We performed 10-fold cross validation within each training data subset
and determined λ that minimizes the average mean squared error over 10 iterations
for each α tested. We chose the (α, λ) pair that minimized the mean-squared error
over the 10 iterations."

## Data structure

1. **FSCV data**: The data exported from HDCV. Each file corresponds to the FSCV response of a fixed concentration or condition (different pH value) of solutes such as DA, NE, or 5-HT. Since we are using Octaflow with 16 channels, there will be 16 distinct files of FSCV data for each session of calibration data.
2. **EN data**: The FSCV data should be preprocessed according to Kishida's paper. A short snippet of data should be extracted from each file of the FSCV data. Also, all the 16 files are concatenated. We will also prepare a corresponding file of the concentration for each sample.
3. **All data**: For each session, there will be exactly one pair of EN data (E##_YYYYMMDD_FSCV.npy and E##_YYYYMMDD_CONC.npy). However, in order to train an EN model, we need multiple sessions of data to maximize the generalizability. Therefore, we concatenate the chosen EN_data as All_data, which still requires further processing to be qualified for EN modeling.
4. **Resampled data**: After concatenation, we need to resample the data so that the concentration in y is normal about a specified value with a specified variance. More details can be found in Kishida paper's SI.

## Workflow
1. **Random_concentration.py**: We use it to determine the concentration profile of that in vitro session.
2. **Octaflow_preprocess.py**: After collecting and exporting in vitro calibration data (aka the FSCV_data), we use it to transform them into EN_data. The preprocessing involves in an optional manual cleaning, which is to take actions for a run, either choosing a snippet other than 75-115 seconds of data or dropping the entire run.
3. **Elastic_net.py**: After processing and exporting multiple npy files in the EN_data files, we are ready to perform EN regression.
