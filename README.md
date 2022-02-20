# Elastic-Net Based Regression Model for Fast-scan Cyclic Voltammetry

This repository is the official implementation of the elastic-net based regression model for fast-scan cyclic voltammetry. This package consists of a series of calibration data in the `Data/EN_data` folder.

## Requirements:
This repository is written in `python 3.7.11`. You will also need:
- `xlwings` 0.24.9
- `sklearn` 0.24.2

## Model description
This package is meant to infer concentrations of neurotransimeters (NTs), including dopamine (DA) and serotonin (5-HT), either *in vitro* or *in vivo*. The detailed methodology is described in [this paper](https://www.pnas.org/content/113/1/200). In brief, the fast-scan cyclic voltammetry (FSCV) response of some known concentrations of NT are measured using a carbon-fiber electrode. The non-background-subtracted FSCV responses and the corresponding concentrations are then stored for later training. To infer concentrations in new recorded data, we have to prepare a brief recording session (more than 40 seconds of steady signal) in phosphate-buffered saline besides the real experimental session. This baseline signal can be seen as the characteristic of the electrode of interest. The baseline signal will be clustered with the electrodes used in the calibration data set in order to select electrodes that are intrinsically similar. Those selected sessions will then be trained by the elastic-net regression algorithm, which would yield a list of models that target a specific concentration range. The `interchanging` algorithm will then constantly select the best model to infer concentration over time.
