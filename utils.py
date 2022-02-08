#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# utils.py

import os
import numpy as np
import pandas as pd
import logging
import Clustering

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)8s] --- %(message)s','%Y-%m-%d %H:%M:%S')
file_handler = logging.FileHandler('Log/action.log')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


data_path  = 'Data/EN_data'
fscv_path  = 'Data/FSCV_data'
model_path = 'Data/EN_model'
eval_path  = 'Data/Model_evaluation'

def ask_page():
    _ = input('Assign page (High DA, Low DA, pH, 5-HT, or NE): ')
    if _ in ['High DA', 'Low DA', 'pH', '5-HT', 'NE']:
        print(f'Assigned page: {_}')
        return _
    else:
        return ask_page()

def check_status(file):
    if os.path.exists(file):
        return True
    else:
        logger.warning(f'Not exist: {file.split("/")[-1]}')
        return False

def prepare(var,page,sessions=[],diff=True,view_session=True):

    calibration = pd.read_excel('Log/calibration_log.xlsx',page)
    calibration['Date'] = calibration['Date'].dt.strftime('%Y%m%d')
    if sessions == []:
        sessions = calibration.index.values

    if var == 'x':
        result = np.empty((0,1000))
        suffix = 'FSCV'
    elif var == 'y':
        result = np.empty((0,))
        suffix = 'CONC'
    else:
        raise ValueError('Wrong input!')

    session_names = []
    for i in range(len(sessions)):
        date      = calibration.iloc[sessions[i]].loc['Date']
        electrode = calibration.iloc[sessions[i]].loc['Electrode']
        file = os.path.join(data_path,f'{electrode}_{date}_{suffix}.npy')
        if check_status(file):
            _ = np.load(file)
            result = np.concatenate((result,_))
            session_names.append(f'{electrode}_{date}')
    if diff == True and var == 'x':
        result = np.diff(result) * 100000 # 100 kHz
    if view_session:
        print('Used sessions:')
        for id,name in zip(sessions,session_names):
            print(f'Session {id}: {name}')
    return result, session_names
