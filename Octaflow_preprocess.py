#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Octaflow_preprocess.py

import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import pandas as pd
import utils

## Folder info
data_path = utils.data_path
fscv_path = utils.fscv_path

## Load calibration log
print('======================')
print('Start preprocessing...')
calibration = utils.load_calibration_log()
# Check status of processing data
for session in range(len(calibration)):
    date      = calibration.iloc[session].loc['Date']
    electrode = calibration.iloc[session].loc['Electrode']
    print(f'Session {session}: {date} - {electrode}')
    raw_file = os.path.join(fscv_path,f'{date}_{electrode}_Octaflow')
    x_file   = os.path.join(data_path,f'{electrode}_{date}_FSCV.npy')
    y_file   = os.path.join(data_path,f'{electrode}_{date}_CONC.npy')
    utils.check_status(raw_file)
    utils.check_status(x_file)
    utils.check_status(y_file)

## The two important parameters of each session
print('==================================')
session = int(input('Which session to be preprocessed: '))
date      = calibration.iloc[session].loc['Date']
electrode = calibration.iloc[session].loc['Electrode']
print('==========================')
print(f' > Date: {date}')
print(f' > Electrode: {electrode}')

## Load files
# Find the corresponding files in the data_folder and extract the raw_data
print('===================')
print('Loading raw data...')
data_folder = f'{date}_{electrode}_Octaflow'
files       = glob.glob(os.path.join(fscv_path,data_folder,f'{electrode}.hdcv*Color.txt'))
raw_data    = [np.loadtxt(files[i]) for i in range(len(files))]
print('Raw data loaded...')

## Show raw data individually at the oxidation peak (data point 300)
# The goal of this visualization is to inspect whether we want to
# keep or discard some runs (or concentrations).
plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.plot(np.arange(0,raw_data[i].shape[1])/10, raw_data[i][300,:],'k',lw=2)
    plt.axvline(75)
    plt.axvline(115)
plt.savefig(os.path.join(fscv_path,data_folder,f'{date}_{electrode}_raw.png'))
plt.show()

## Auto cleaning
# Automatic cleaning that works for all runs
print('=====================')
print('Automatic cleaning...')
conc     = calibration.iloc[session].loc[1:].to_numpy(dtype='int')
conc     = np.append(0,conc)
data_mod = [datum[:,750:1150] for datum in raw_data]

## Manual cleaning
print('==================')
print('Manual cleaning...')
while int(input('Manual cleaning or not (0/1)? ')):

    print('=================')
    print('Start cleaning...')
    run    = int(input('Which run to clean (zero-indexed)? '))
    action = input('Which action: fix or drop? ')

    if action.startswith('f'):
        start = input('What is the starting point of the 400 data points? ')
        data_mod[run] = raw_data[run][:,start:start+400]
    elif action.startswith('d'):
        data_mod.pop(run)
        conc = np.delete(conc, run)
    else:
        print('Invalid action.')

## Reconstruct data
print('======================')
print('Reconstructing data...')
data  = np.concatenate(data_mod,axis=1)
concs = np.repeat(conc,400)

## Save files
if int(input('Sure to save (0/1)? ')):
    if data.shape[1] == concs.shape[0]: # The final check
        print('==============')
        print('Saving data...')
        np.save(os.path.join(data_path,f'{electrode}_{date}_FSCV.npy'),data.T)
        np.save(os.path.join(data_path,f'{electrode}_{date}_CONC.npy'),concs)
    else:
        print('Error!')
else:
    print('Check again and restart.')
