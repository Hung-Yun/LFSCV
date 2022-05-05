# Elastic-Net Based Regression Model for Fast-scan Cyclic Voltammetry

This repository is the official implementation of the elastic-net based regression model for fast-scan cyclic voltammetry. This package consists of a series of calibration data in the `Data/EN_data` folder.

## Requirements:
This repository is written in `python 3.7.11`. You will also need:
- `xlwings` 0.24.9
- `sklearn` 0.24.2

## Model description
This package is meant to infer concentrations of neurotransimeters (NTs), including dopamine (DA) and serotonin (5-HT), either *in vitro* or *in vivo*. The detailed methodology is described in [this paper](https://www.pnas.org/content/113/1/200). In brief, the fast-scan cyclic voltammetry (FSCV) response of some known concentrations of NT are measured using a carbon-fiber electrode. The non-background-subtracted FSCV responses and the corresponding concentrations are then stored for later training. To infer concentrations in new recorded data, we have to prepare a brief recording session (more than 40 seconds of steady signal) in phosphate-buffered saline besides the real experimental session. This baseline signal can be seen as the characteristic of the electrode of interest. The baseline signal will be clustered with the electrodes used in the calibration data set in order to select electrodes that are intrinsically similar. Those selected sessions will then be trained by the elastic-net regression algorithm (`ElasticNet`), which would yield a list of models that target a specific concentration range. Our algorithm (`interchanging`) will then constantly select the best model to infer concentration over time. (See figure below).

![Illustration](https://github.com/Hung-Yun/EN_FSCV/blob/main/FSCV%20illustration.png)

## General workflow

### Preprocessing data


## Contact
File any issues with the [issue tracker](https://github.com/Hung-Yun/EN_FSCV/issues). For any questions or problems, please contact [Hung-Yun Lu](https://github.com/Hung-Yun).

## Reference
- Kishida, K. T., Saez, I., Lohrenz, T., Witcher, M. R., Laxton, A. W., Tatter, S. B., White, J. P., Ellis, T. L., Phillips, P. E., & Montague, P. R. (2016). [Subsecond dopamine fluctuations in human striatum encode superposed error signals about actual and counterfactual reward](https://doi.org/10.1073/pnas.1513619112). Proceedings of the National Academy of Sciences of the United States of America, 113(1), 200–205.
- Montague, P. R., & Kishida, K. T. (2018). [Computational Underpinnings of Neuromodulation in Humans](https://doi.org/10.1101/sqb.2018.83.038166). Cold Spring Harbor symposia on quantitative biology, 83, 71–82.
