#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# utils.py

import os
import numpy as np
import pandas as pd
import logging

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
        logger.info(f'Assigned page: {_}')
        return _
    else:
        return ask_page()

def load_calibration_log():
    page = ask_page()
    logger.info('Load in calibration_log.xlsx.')
    calibration = pd.read_excel('Log/calibration_log.xlsx',page)
    calibration['Date'] = calibration['Date'].dt.strftime('%Y%m%d')
    return calibration

def check_status(file):
    if os.path.exists(file):
        return True
    else:
        logger.warning(f'Not exist: {file.split("/")[-1]}')
        return False

def prepare(var):
    logger.info(f'Now preparing {var}')
    calibration = load_calibration_log()
    if var == 'x':
        result = np.empty((0,1000))
        suffix = 'FSCV'
    elif var == 'y':
        result = np.empty((0,))
        suffix = 'CONC'
    else:
        raise ValueError('Wrong input!')
    used_sessions = []
    # Current default: use all sessions.
    # Future: decide sessions after clustering.
    for session in range(len(calibration)):
        date      = calibration.iloc[session].loc['Date']
        electrode = calibration.iloc[session].loc['Electrode']
        logger.info(f'Session {session}: {date} - {electrode}.')
        file = os.path.join(data_path,f'{electrode}_{date}_{suffix}.npy')
        if check_status(file):
            _ = np.load(file)
            result = np.concatenate((result,_))
            used_sessions.append([f'{electrode}_{date}'])
    if var == 'x':
        result = np.diff(result) * 100000
    return result, used_sessions
