#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# utils.py

import os
import numpy as np
import pandas as pd

box_path    = '/Users/hungyunlu/Library/CloudStorage/Box-Box'
note_path   = os.path.join(box_path,'Hung-Yun Lu Research File/Projects/_LabNote')
script_path = os.path.join(box_path,'Hung-Yun Lu Research File/Projects/FSCV/Script')
data_path   = os.path.join(script_path,'Data/EN_data')
fscv_path   = os.path.join(script_path,'Data/FSCV_data')
model_path  = os.path.join(script_path,'Data/EN_model')
eval_path   = os.path.join(script_path,'Data/EN_model/Model_evaluation')

if os.getcwd() != script_path:
    os.chdir(script_path)
    print('================================')
    print(f'Dirpath change to: {script_path}')

def ask_page():
    _ = input('Assign page (High DA, Low DA, pH, 5-HT, or NE): ')
    if _ in ['High DA', 'Low DA', 'pH', '5-HT', 'NE']:
        print(f' > Assigned page: {_}')
        return _
    else:
        return ask_page()

def load_calibration_log():
    page = ask_page()
    calibration = pd.read_excel(os.path.join(note_path,'_Calibration_log.xlsx'),page)
    calibration['Date'] = calibration['Date'].dt.strftime('%Y%m%d')
    return calibration

def check_status(file):
    if os.path.exists(file):
        return True
    else:
        print(f' > Not exist: {file.split("/")[-1]}')
        return False

def prepare(var):
    print(f' > Now preparing {var}')
    calibration = load_calibration_log()
    if var == 'x':
        result = np.empty((0,1000))
        suffix = 'FSCV'
    elif var == 'y':
        result = np.empty((0,))
        suffix = 'CONC'
    else:
        raise ValueError('Wrong input!')
    print('================================')
    print(f'Start preparing {var}')
    used_sessions = []
    for session in range(len(calibration)):
        date      = calibration.iloc[session].loc['Date']
        electrode = calibration.iloc[session].loc['Electrode']
        print(f'Session {session}: {date} - {electrode}.')
        file = os.path.join(data_path,f'{electrode}_{date}_{suffix}.npy')
        if check_status(file):
            _ = np.load(file)
            result = np.concatenate((result,_))
            used_sessions.append([f'{electrode}_{date}'])
    if var == 'x':
        result = np.diff(result) * 100000
    return result, used_sessions
