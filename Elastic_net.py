#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Elastic_net.py

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle
import pickle
import utils
import xlwings as xw
import Clustering

import datetime
today = datetime.date.today().strftime('%y%m%d')

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)8s] --- %(message)s','%Y-%m-%d %H:%M:%S')
file_handler = logging.FileHandler('Log/EN.log')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.info(f'-- Now running {os.path.basename(__file__)} --')

## Folder info
model_path = utils.model_path
eval_path  = utils.eval_path

## Prepare x and y for regression
index = Clustering.cluster()
page  = utils.ask_page()
x, sessions = utils.prepare('x',page,index,view_session=True)
y, sessions = utils.prepare('y',page,index,view_session=False)

## Visualize distribution
if int(input('Visualize distribution or not (0/1)? ')):
    a,b = np.unique(y,return_counts=True)
    plt.figure(figsize=(10,8))
    plt.bar(a.astype(int).astype(str),b)
    plt.ylabel('Count')
    plt.xlabel('Concentration (nM)')
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(eval_path,f'Dist-{today}.png'))
    logger.info(f'Distribution saved as Dist-{today}.png.')
else:
    logger.info(f'Distribution not saved as Dist-{today}.png.')

## Prepare subsampling
# Input the concentration profile
mu    = int(input(' > The centered concentration: '))
sigma = int(input(' > The variance of the distribution: '))
size  = int(input(' > The amount of samples in total: '))
base  = int(input(' > Round up base: '))
model_name = f'Model-{mu}-{sigma}-{size}-{today}'
logger.info(f'Concentration profile:')
logger.info(f'Centered conc: {mu}.')
logger.info(f'Variance: {sigma}.')
logger.info(f'Sample size: {size}.')
logger.info(f'Model name: Model-{mu}-{sigma}-{size}-{today}')

# Normal about the target concentration, and round to the base
# (in the high DA situation, 50 nM).
s = np.random.normal(mu, sigma, size)
myround = lambda x: base*round(x/base)
round_s = [myround(i) for i in s]
sub_conc, counts = np.unique(round_s, return_counts=True)

# Concatenating samples from each sub_conc
xx = np.empty((0,999))
yy = np.empty((0))
logger.info('Start resampling.')
for i in range(len(sub_conc)):
    if sum(y==sub_conc[i]) != 0:
        sub_x = x[ y==sub_conc[i] , : ]
        sub_y = y[ y==sub_conc[i] ]
        if sum(y==sub_conc[i]) > counts[i]:
            np.random.seed(1)
            ids = np.random.choice(len(sub_x),counts[i],replace=False)
            xx  = np.concatenate((xx,sub_x[ids,:]))
            yy  = np.append(yy,sub_y[ids])
        else:
            xx  = np.concatenate((xx,sub_x))
            yy  = np.append(yy,sub_y)
    else:
        logger.warning(f'No samples for {sub_conc[i]}.')

# Optional histogram
a,b = np.unique(yy,return_counts=True)
if int(input('Plot histogram (0/1)? ')):
    plt.bar(a.astype(int).astype(str),b)
    plt.xlabel('Concentration (nM) or pH')
    plt.ylabel('Counts')
    plt.xticks(rotation=90)
    plt.show()

## Fit elastic net regression
xx, yy = shuffle(xx, yy, random_state=0)
x_train,x_test,y_train,y_test = train_test_split(xx, yy, test_size=0.2)
l1_ratios = np.arange(1,0,-0.1)
cv = KFold(n_splits=10, shuffle=True)
model = ElasticNetCV(l1_ratio=l1_ratios, cv=cv, n_jobs=-1)
logger.info('Start fitting model')
model.fit(x_train, y_train)
logger.info(f'Finish model fitting - alpha: {model.alpha_:.2f}; l1_ratio: {model.l1_ratio_:.2f}')

## Store model
savefile = os.path.join(model_path,f'{model_name}.sav')
pickle.dump(model,open(savefile,'wb'))
logger.info(f'Model saved as {model_name}.sav.')

# Save tested data so that they can be evaluated separately in Model_evaluation.py
np.save(os.path.join(model_path,f'{model_name}-xtest.npy'), x_test)
np.save(os.path.join(model_path,f'{model_name}-ytest.npy'), y_test)

# Save trained data so that they can be concatenated with other train data for other models
np.save(os.path.join(model_path,f'{model_name}-xtrain.npy'), x_train)
np.save(os.path.join(model_path,f'{model_name}-ytrain.npy'), y_train)

logger.info('Train/test data saved.')

df = pd.read_pickle('Log/EN.pkl')
df.loc[f'{model_name}'] = [mu, sigma, size, today, np.array([a,b]), sessions]
df.to_pickle('Log/EN.pkl')
logger.info('Model info saved in EN.pkl.')
