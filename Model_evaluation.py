#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Model_evaluation.py

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

## Load model
wb = xw.Book(os.path.join(model_path,'Model_log.xlsx'))
sheet = wb.sheets['List']
models = sheet.range('A1').expand().value
print('================================')
print('All models:')
for i,model in enumerate(models):
    print(f'Model {i}: {model}')

model_name = model_names[int(input('Evaluate which model (indicate model number): '))]
model = pickle.load(open(model_name, 'rb'))
model_name = model_name.split("/")[-1]
print(f'\nNow evaluating: {model_name}\n')

wb = xw.Book(os.path.join(model_path,'Model_log.xlsx'))
sheet = wb.sheets[f'{model_name.split(".")[0]}']
samples = np.array(sheet.range('A5').expand().value)
sample_conc = samples[:,0]
sample_count = samples[:,1]

# ## Load data
# x = utils.prepare('x')
# y = utils.prepare('y')
#
# print('>>>>>>>>>>>>')
# print(xx.shape)
# print(yy.shape)

## SNR
'''

print('================')
print('Evaluate the SNR')

goals = np.array(np.unique(y,return_counts=True),dtype=int)[0]
SNR = np.zeros((len(goals),2))
SNR[:,0] = goals

for goal in goals:
    pred    = mod.predict(x[y==goal])
    epsilon = pred - goal
    snr = (goal/np.sqrt(np.sum(epsilon**2)/sum(y==goal)))**2
    print(f'{goal} SNR: {snr:.2f}')
    SNR[SNR[:,0]==goal,1] = snr

'''


## Visualze performance









def visualize_performance(mod, x, y):

    Truth = y[:,np.newaxis]
    Label = model.predict(x)[:,np.newaxis]
    Diff  = Truth - Label
    concs = np.array(np.unique(y,return_counts=True),dtype=int)

    Conc = list(map(str,(concs[0])))
    Mean = [abs(np.mean(Diff[Truth==s])) for s in concs[0]]
    Std  = [np.std(Diff[Truth==s])/len(Diff[Truth==s]) for s in concs[0]]

    x_pos = np.arange(len(Conc))
    plt.bar(x_pos, Mean,align='center', alpha=0.5, ecolor='black', capsize=2)
    plt.errorbar(x_pos, Mean, yerr=Std,ls='none')
    plt.ylabel('Difference between truth and prediction (nM)')
    plt.xticks(x_pos,Conc)
    plt.xlabel('True concentration (nM)')
    plt.xticks(rotation=90)
    plt.axhline(5)
    plt.show()
