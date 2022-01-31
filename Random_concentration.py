#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# random_concentration.py

import os
import numpy as np
import xlwings as xw
import utils

import datetime
today = datetime.date.today().strftime('%Y-%m-%d')

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
logger.info(f'-- Now running {os.path.basename(__file__)} --')

## Concentration info
high_DA = np.arange(1500,-1,-50)
low_DA  = np.arange(150,0,-5)
seq = np.random.permutation(30)[:15]

inputs = {'High DA': high_DA,
          'Low DA': low_DA}
page = utils.ask_page()
conc = np.sort(inputs[page][seq])
logger.info(f'Concentrations: {conc}.')

# Save info into _Calibration_log file
if int(input('Write in excel file (0/1)? ')):
    logger.info('Write in calibration_log.xlsx.')
    wb = xw.Book('Log/calibration_log.xlsx')
    sheet = wb.sheets[page]
    row = str(sheet.range('A' + str(sheet.cells.last_cell.row)).end('up').row + 1)
    sheet.range('A'+row).value = today
    sheet.range('B'+row).value = input('Which electrode? ')
    sheet.range('C'+row).value = conc
    wb.save()
    logger.info(f'Electrode: {sheet.range("B"+row).value}.')
else:
    logger.warning('Not written in calibration_log.xlsx.')
