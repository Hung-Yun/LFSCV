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
3. **All data**: For each session, there will be exactly
