'''
Infer neurotransmitter's concentrations from in vitro or in vivo recordings using
fast-scan cyclic voltammetry. This algorithm constantly investigate which model
predicts the most accurate result.
'''

import ElasticNet as ENet
import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pred_path = ENet.pred_path
model_path = ENet.model_path

def load_exp(filename):
    '''
    Load the file we'd like to predict the concentration.
    The loaded file should be in the shape of (N x 1000).
    Before returning, the function will also preprocess it into (N x 999).

    INPUT
        1. filename: str, the name of the file in the folder of Data/EN_data.

    OUTPUT
        1. data: array (N x 999).
    '''
    filepath = os.path.join(pred_path, filename)
    if ENet.check_status(filepath):
        data = np.load(filepath)
        if data.shape[1] != 1000:
            raise ValueError('Wrong shape of file.')
        return np.diff(data) * 100000

def WRONG_load_models(target):
    '''
    Load the models specific to the electrode used for the exp file.

    INPUT
        1. target: str, the target session

    OUTPUT
        1. model_dict: dict, {conc: model}
    '''

    files = glob.glob(os.path.join(model_path,target,'*.sav'))
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    model = [pickle.load(open(file,'rb')) for file in files]
    conc  = [int(file.split("_")[-1].split(".")[0]) for file in files]
    model_dict = {key:value for key in conc for value in model}
    return model_dict


# def pred(sample):
#     model = models[0]
#     threshold  = [concs[0]+3*sigmas[0], concs[0]-3*sigmas[0]]
#     prediction = model.predict(sample.reshape(1,-1))
#     if prediction>threshold[0] or prediction<threshold[1]:
#         models[0], models[1] = models[1], models[0]
#         concs[0],  concs[1]  = concs[1],  concs[0]
#         sigmas[0], sigmas[1] = sigmas[1], sigmas[0]
#         return pred(sample)
#     else:
#         return prediction
#
# low_conc, low_sigma = model_names[low_id].split("-")[1:3]
# high_conc, high_sigma = model_names[high_id].split("-")[1:3]
# concs  = [int(low_conc),  int(high_conc)]
# sigmas = [int(low_sigma), int(high_sigma)]
# y_pred_sim = np.empty((0))
# for i in range(x_test_sim.shape[0]):
#     y_pred_sim = np.append(y_pred_sim,pred(x_test_sim[i,:]))
