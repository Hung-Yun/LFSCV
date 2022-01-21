#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# random_concentration.py

import os
import numpy as np
import xlwings as xw
import utils

import datetime
today = datetime.date.today().strftime('%Y-%m-%d')

## Folder info
note_path = utils.note_path

## Concentration info
high_DA = np.arange(1500,-1,-50)
low_DA  = np.arange(150,0,-5)
seq = np.random.permutation(30)[:15]

inputs = {'High DA': high_DA,
          'Low DA': low_DA}
page = utils.ask_page()
conc = np.sort(inputs[page][seq])

# Save info into _Calibration_log file
wb = xw.Book(os.path.join(note_path,'_Calibration_log.xlsx'))
sheet = wb.sheets[page]
row = str(sheet.range('A' + str(sheet.cells.last_cell.row)).end('up').row + 1)
sheet.range('A'+row).value = today
sheet.range('B'+row).value = input('Which electrode? ')
sheet.range('C'+row).value = conc
wb.save()
