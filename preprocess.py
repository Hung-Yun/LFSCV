#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Octaflow_preprocess.py

import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import pandas as pd
import ElasticNet as EN
import argparse

## Folder and file info
data_path   = EN.data_path
fscv_path   = EN.fscv_path
eval_path   = EN.eval_path
pred_path   = EN.pred_path
calibration = EN.calibration

def experiment():

    # ------ Preprocess clustering data ------

    ## Check status of processing data
    session_list = []
    for session in range(len(calibration)):
        if calibration.Analyte[session] not in {'DA','5-HT','pH'}:
            print(f'Session {session}:\t {calibration.Session[session]}')
            EN.check_status(os.path.join(fscv_path,calibration.FSCV_data[session]))
            EN.check_status(os.path.join(data_path,calibration.Session[session]+'_FSCV.npy'))
            EN.check_status(os.path.join(data_path,calibration.Session[session]+'_CONC.npy'))
            session_list.append(session)

    ## Load files
    # Find the corresponding files in the data_folder and extract the raw_data
    session  = int(input('Which session to be preprocessed: '))
    if session not in session_list:
        raise ValueError('Session not in the correct experimental type')
    data     = calibration.iloc[session]
    raw_data = np.loadtxt(os.path.join(fscv_path,data.FSCV_data,f'Cluster.hdcv Color.txt'))

    ## Construct x (FSCV) and y (CONC)
    FSCV = raw_data[:,750:1150]
    CONC = np.zeros((400,))

    ## Save files
    if FSCV.shape[1] == CONC.shape[0]: # The final check
        np.save(os.path.join(data_path,f'{data.Session}_FSCV.npy'),FSCV.T)
        np.save(os.path.join(data_path,f'{data.Session}_CONC.npy'),CONC)
        print(f'Data saved in {data.Session}_FSCV.npy and {data.Session}_CONC.npy')
    else:
        print('Sizes do not match.')

    # ------ Preprocess experimental data ------

    while True:
        exp_name = input('The name of the experiment: ')
        if len(exp_name) == 0:
            break
        files    = glob.glob(os.path.join(fscv_path,data.FSCV_data,f'{exp_name}.hdcv*Color.txt'))
        if len(files) > 1:
            files.sort(key=lambda x: int(x.split("/")[-1].split()[0].replace(f'{exp_name}.hdcv','')))
        raw_data = [np.loadtxt(files[i]) for i in range(len(files))]
        FSCV = np.concatenate(raw_data,axis=1)
        np.save(os.path.join(pred_path,f'{data.Session}_{exp_name}.npy'),FSCV.T)
        print(f'Data saved in {data.Session}_{exp_name}.npy')


def octaflow():

    # ------ Preprocess Octaflow data ------

    ## Check status of processing data
    session_list = []
    for session in range(len(calibration)):
        if calibration.Analyte[session] in {'DA','5-HT','pH'}:
            print(f'Session {session}:\t {calibration.Session[session]}')
            EN.check_status(os.path.join(fscv_path,calibration.FSCV_data[session]))
            EN.check_status(os.path.join(data_path,calibration.Session[session]+'_FSCV.npy'))
            EN.check_status(os.path.join(data_path,calibration.Session[session]+'_CONC.npy'))
            session_list.append(session)

    ## Load files
    # Find the corresponding files in the data_folder and extract the raw_data
    while True:
        session  = int(input('Which session to be preprocessed: '))
        if session < 0:
            break
        if session not in session_list:
            raise ValueError('Session not in the correct experimental type')
        data     = calibration.iloc[session]
        files    = glob.glob(os.path.join(fscv_path,data.FSCV_data,f'{data.Electrode}.hdcv*Color.txt'))
        files.sort(key=lambda x: int(x.split("/")[-1].split()[0].replace(f'{data.Electrode}.hdcv','')))
        raw_data = [np.loadtxt(files[i]) for i in range(len(files))]

        ## Show raw data individually at the oxidation peak (data point 300)
        # The goal of this visualization is to inspect whether we want to
        # keep or discard some runs (or concentrations).
        plt.figure(figsize=(10,10))
        plt.subplots_adjust(hspace=0.3,wspace=0.3)
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.plot(np.arange(0,raw_data[i].shape[1])/10, raw_data[i][300,:],'k',lw=2)
            plt.yticks(rotation=60)
            plt.title(f'Run {i}',color='r')
            plt.axvline(75,c='b',ls='--',lw=1.5)
            plt.axvline(115,c='b',ls='--',lw=1.5)
        plt.savefig(os.path.join(eval_path,'Preprocess',f'Preprocess-{data.Session}.png'))
        print(f'Raw data saved as Preprocess-{data.Session}.png')

        ## Construct x (FSCV) and y (CONC)
        conc = np.zeros((16,))
        if data.Analyte != 'pH':
            conc[1:] = data.loc[1:15]

        ## Cleaning
        fscv_mod = [datum[:,750:1150] for datum in raw_data]
        while int(input('Manual cleaning or not (0/1)? ')):
            run = int(input('Which run to discard (zero-indexed, start from greater number)? '))
            fscv_mod.pop(run)
            conc = np.delete(conc, run, axis=0)
        print('Finish manual cleaning.')

        ## Reconstruct data
        FSCV = np.concatenate(fscv_mod,axis=1)
        CONC = np.repeat(conc,400)

        ## Save files
        if FSCV.shape[1] == CONC.shape[0]: # The final check
            np.save(os.path.join(data_path,f'{data.Session}_FSCV.npy'),FSCV.T)
            np.save(os.path.join(data_path,f'{data.Session}_CONC.npy'),CONC)
            print(f'Data saved in {data.Session}_FSCV.npy and {data.Session}_CONC.npy')
        else:
            print('Sizes do not match.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type',type=str,choices=['octaflow','experiment'],
        help='Determine whether to process the calibration data or experimental data.')
    args = parser.parse_args()
    octaflow() if args.type == 'octaflow' else experiment()
