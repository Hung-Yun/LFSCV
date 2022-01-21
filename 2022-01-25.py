#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 2022-01-25.py


'''

Goal:
Try to see whether a combined model is better than a model that is trained separately.
Also compare the parameters (mu, sigma, and size)

'''


# Part of unfinished Model_evaluation.py

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils
import xlwings as xw

## Folder info
model_path = utils.model_path
eval_path  = utils.eval_path

## Read how many models are there
wb = xw.Book(os.path.join(model_path,'Model_log.xlsx'))
sheet = wb.sheets['List']
models = sheet.range('A1').expand().value
# print('================================')
# print('All models:')
# for i,model in enumerate(models):
#     print(f'Model {i}: {model}')

## Ask which models to use
# UNFINISHED
# def ask_model():
#     model_index = []
#     id = int(input('Which model to use? '))
#     if id < len(models):
#         model_index.append(id)
#         return ask_model()
#     else:
#         return model_index
#
# IDs = ask_model()
# print(IDs)
# print(type(IDs))
IDs = [4,2]
models = [models[i] for i in IDs]

## Visualize what their distributions look like


## Load individual models.


## Load x_train, y_train. Make combined model


## Load x_test, y_test. Concatenate them. Use the combined model to test with it.


## Simulated response: Rearrange x_test and y_test so they look like phasic and tonic responses.


## Use individual models and combined model to infer the concentration.


## Visualize them and report MSE for each model.
