#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 2022-01-25.py

'''
Goal:
Try to see whether a combined model is better than a model that is trained separately.
'''

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils
import xlwings as xw
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse

import warnings
warnings.filterwarnings("ignore")

## Folder info
model_path = utils.model_path
eval_path  = utils.eval_path

## Index to prevent overwriting files
ind = input('Index for figures or models (e.g., "-2")')


## Read how many models are there
wb = xw.Book(os.path.join(model_path,'Model_log.xlsx'))
sheet = wb.sheets['List']
model_names = sheet.range('A1').expand().value
print('================================')
print('All models:')
for i,model in enumerate(model_names):
    print(f'Model {i}: {model}')

## Ask which models to use
low_id  = int(input('Which model for Low DA? '))  # 4, 6
high_id = int(input('Which model for High DA? ')) # 2, 5
IDs = [low_id,high_id]
models = [model_names[i] for i in IDs]

## Visualize what their distributions look like
def plot_dist(model):
    sheet = wb.sheets[model]
    dist  = np.array(sheet.range('A5').expand().value)
    index = [str(int(i)) for i in dist[:,0]]
    value = dist[:,1]
    plt.bar(index,value)
    plt.xlabel('Concentration (nM)')
    plt.ylabel('Count')
    for i,v in enumerate(value):
        plt.text(i, v+5, str(int(v)), ha='center',fontdict=dict(fontsize=8))

def save(path,filename):
    counter = 1
    while os.path.exists(os.path.join(path,filename)):
        counter += 1
        filename += str(counter)
    return os.path.join(path,filename)

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.title('Model for low DA')
plot_dist(models[0])
plt.subplot(122)
plt.title('Model for high DA')
plot_dist(models[1])
plt.tight_layout()
plt.savefig(os.path.join(eval_path, f'2022-01-25-Figure1{ind}.png'))
plt.clf()

## Load x_train, y_train. Make combined model
## Commented out after the combined model is trained and saved
# x_train_low  = np.load(os.path.join(model_path,f'{models[0]}-xtrain.npy'))
# y_train_low  = np.load(os.path.join(model_path,f'{models[0]}-ytrain.npy'))
# x_train_high = np.load(os.path.join(model_path,f'{models[1]}-xtrain.npy'))
# y_train_high = np.load(os.path.join(model_path,f'{models[1]}-ytrain.npy'))
# x_train_mix  = np.concatenate((x_train_low, x_train_high))
# y_train_mix  = np.concatenate((y_train_low, y_train_high))
# l1_ratios = np.arange(1,0,-0.1)
# cv = KFold(n_splits=10, shuffle=True)
# m_mix = ElasticNetCV(l1_ratio=l1_ratios, cv=cv, n_jobs=-1)
# print('================================')
# print('Fitting model...')
# m_mix.fit(x_train_mix, y_train_mix)
# print('Finish training...')
# savefile = os.path.join(model_path,f'2022-01-25-mix{ind}.sav')
# pickle.dump(m_mix,open(savefile,'wb'))

## Load individual models.
m_low  = pickle.load(open(os.path.join(model_path,f'{models[0]}.sav'),'rb'))
m_high = pickle.load(open(os.path.join(model_path,f'{models[1]}.sav'),'rb'))
m_mix  = pickle.load(open(os.path.join(model_path,f'2022-01-25-mix{ind}.sav'),'rb'))

## Load x_test, y_test. Concatenate them. Use the combined model to test with it.
x_test_low  = np.load(os.path.join(model_path,f'{models[0]}-xtest.npy'))
y_test_low  = np.load(os.path.join(model_path,f'{models[0]}-ytest.npy'))
x_test_high = np.load(os.path.join(model_path,f'{models[1]}-xtest.npy'))
y_test_high = np.load(os.path.join(model_path,f'{models[1]}-ytest.npy'))
x_test_mix  = np.concatenate((x_test_low, x_test_high))
y_test_mix  = np.concatenate((y_test_low, y_test_high))
ms      = [m_low, m_high, m_mix]
x_tests = [x_test_low, x_test_high, x_test_mix]
y_tests = [y_test_low, y_test_high, y_test_mix]
result = np.zeros((3,))
for i in range(3):
    result[i] = ms[i].score(x_tests[i],y_tests[i])

plt.figure()
plt.bar(['Low DA','High DA','Combined'], result)
plt.title('Performance of each model (tested with unseen data)')
plt.ylabel('Performance')
plt.ylim([0.5,0.9])
plt.savefig(os.path.join(eval_path, f'2022-01-25-Figure2{ind}.png'))
plt.clf()


## Simulated response: Rearrange x_test and y_test so they look like phasic and tonic responses.
np.random.seed(1)
low_index  = np.random.choice(y_test_low.shape[0], y_test_low.shape[0], replace=False)
high_index = np.random.choice(y_test_high.shape[0], 200, replace=False)
y_test_sim = y_test_low[low_index]
x_test_sim = x_test_low[low_index,:]
y_test_sim = np.insert(y_test_sim, 200, y_test_high[high_index])
x_test_sim = np.insert(x_test_sim, 200, x_test_high[high_index,:],axis=0)
plt.figure()
plt.plot(y_test_sim,'k',label='Simulated DA response')
plt.ylim([-100,2000])
plt.xlabel('Time (sample)')
plt.ylabel('Concentration (nM)')
plt.title('Simulated DA response and prediction from models')
plt.legend(prop={'size': 8})
plt.savefig(os.path.join(eval_path, f'2022-01-25-Figure3{ind}.png'))
plt.clf()

## Use individual models and combined model to infer the concentration.
def pred(sample):
    model = models[0]
    threshold  = [concs[0]+3*sigmas[0], concs[0]-3*sigmas[0]]
    prediction = model.predict(sample.reshape(1,-1))
    if prediction>threshold[0] or prediction<threshold[1]:
        models[0], models[1] = models[1], models[0]
        concs[0],  concs[1]  = concs[1],  concs[0]
        sigmas[0], sigmas[1] = sigmas[1], sigmas[0]
        return pred(sample)
    else:
        return prediction

low_conc, low_sigma = model_names[low_id].split("-")[1:3]
high_conc, high_sigma = model_names[high_id].split("-")[1:3]
concs  = [int(low_conc),  int(high_conc)]
sigmas = [int(low_sigma), int(high_sigma)]
y_pred_sim = np.empty((0))
for i in range(x_test_sim.shape[0]):
    y_pred_sim = np.append(y_pred_sim,pred(x_test_sim[i,:]))
plt.figure()
plt.plot(m_low.predict(x_test_sim),'g',alpha=0.6,label='Low DA')
plt.plot(m_high.predict(x_test_sim),'r',alpha=0.6,label='High DA')
plt.plot(m_mix.predict(x_test_sim),'b',alpha=0.6,label='Combined')
plt.plot(y_pred_sim,'k',alpha=0.6,label='Interchanging')
plt.legend(prop={'size': 8})
plt.ylim([-100,2000])
plt.xlabel('Time (sample)')
plt.ylabel('Concentration (nM)')
plt.title('Simulated DA response and prediction from models')
plt.savefig(os.path.join(eval_path, f'2022-01-25-Figure4{ind}.png'))
plt.clf()

## Visualize them and report MSE for each model.
MSE = [mse(y_test_sim,m_low.predict(x_test_sim)),
       mse(y_test_sim,m_high.predict(x_test_sim)),
       mse(y_test_sim,m_mix.predict(x_test_sim)),
       mse(y_test_sim,y_pred_sim)]
plt.figure()
plt.bar(['Low DA','High DA','Combined','Interchanging'], MSE)
plt.title('MSE of each model')
plt.yscale('log')
plt.savefig(os.path.join(eval_path, f'2022-01-25-Figure5{ind}.png'))
plt.clf()
