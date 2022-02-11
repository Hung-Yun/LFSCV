#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Octaflow_preprocess.py

import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import pandas as pd
import ElasticNet

## Folder info
data_path = ElasticNet.data_path
fscv_path = ElasticNet.fscv_path
eval_path = ElasticNet.eval_path
cal_path  = ElasticNet.cal_path

## Load calibration log
print('============')
while True:
    page = input('Which page? ')
    if page not in pd.read_excel(cal_path,None).keys():
        print('Wrong input. No such page.')
    else:
        break
calibration = pd.read_excel(cal_path,page)
calibration['Date'] = calibration['Date'].dt.strftime('%Y%m%d')

# Check status of processing data
for session in range(len(calibration)):
    date      = calibration.iloc[session].loc['Date']
    electrode = calibration.iloc[session].loc['Electrode']
    print(f'Session {session}: {date} - {electrode}')
    ElasticNet.check_status(os.path.join(fscv_path,f'{date}_{electrode}_Octaflow'))
    ElasticNet.check_status(os.path.join(data_path,f'{electrode}_{date}_FSCV.npy'))
    ElasticNet.check_status(os.path.join(data_path,f'{electrode}_{date}_CONC.npy'))

## The two important parameters of each session
session   = int(input('Which session to be preprocessed: '))
date      = calibration.iloc[session].loc['Date']
electrode = calibration.iloc[session].loc['Electrode']

## Load files
# Find the corresponding files in the data_folder and extract the raw_data
data_folder = f'{date}_{electrode}_Octaflow'
files       = glob.glob(os.path.join(fscv_path,data_folder,f'{electrode}.hdcv*Color.txt'))
raw_data    = [np.loadtxt(files[i]) for i in range(len(files))]
print(f'Raw data loaded from {data_folder}')

## Show raw data individually at the oxidation peak (data point 300)
# The goal of this visualization is to inspect whether we want to
# keep or discard some runs (or concentrations).
plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.plot(np.arange(0,raw_data[i].shape[1])/10, raw_data[i][300,:],'k',lw=2)
    plt.axvline(75)
    plt.axvline(115)
plt.savefig(os.path.join(eval_path,f'Preprocess-{electrode}-{date}.png'))
print(f'Raw data saved as Preprocess-{electrode}-{date}.png')

## Auto cleaning
# Automatic cleaning that works for all runs
conc     = calibration.iloc[session].loc[1:].to_numpy(dtype='int')
conc     = np.append(0,conc)
data_mod = [datum[:,750:1150] for datum in raw_data]

## Manual cleaning
while int(input('Manual cleaning or not (0/1)? ')):
    run    = int(input('Which run to clean (zero-indexed)? '))
    action = input('Which action: fix or drop? ')
    if action.startswith('f'):
        start = int(input('What is the starting point of the 400 data points? '))
        data_mod[run] = raw_data[run][:,start:start+400]
        print(f'Manual cleaning run {run}: new starting point from {start}.')
    elif action.startswith('d'):
        data_mod.pop(run)
        conc = np.delete(conc, run)
        print(f'Manual cleaning run {run}: drop run.')
    else:
        print('Invalid action, pick fix/drop.')
print('Finish manual cleaning.')

## Reconstruct data
data  = np.concatenate(data_mod,axis=1)
concs = np.repeat(conc,400)

## Save files
if data.shape[1] == concs.shape[0]: # The final check
    np.save(os.path.join(data_path,f'{electrode}_{date}_FSCV.npy'),data.T)
    np.save(os.path.join(data_path,f'{electrode}_{date}_CONC.npy'),concs)
    print(f'Data saved in {electrode}_{date}_FSCV.npy and {electrode}_{date}_CONC.npy')
else:
    print('Sizes do not match.')
