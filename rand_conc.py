'''
Generate random concentrations for in vitro FSCV calibration experiments.
Write into the excel file 'calibration_log.xlsx'.

For different pH, input manually since the measured pH values are different everytime.
'''

import pandas as pd
import xlwings as xw
import numpy as np
import os
import datetime
import sys

cal_path   = 'Log/calibration_log.xlsx'

conc_range = {
            'High_DA': np.arange(1500,-1,-50),
            'Low_DA':  np.arange(150,0,-5),
            '5-HT':    np.arange(1500,-1,-50)
              }

if len(sys.argv) < 3:
    raise ValueError('Should include which experiment and electrode')
elif len(sys.argv) > 3:
    raise ValueError('Too many inputs')
else:
    if sys.argv[1] not in conc_range.keys():
        raise ValueError('Wrong experiment input')
    else:
        exp, electrode = sys.argv[1], sys.argv[2]

if exp in {'High_DA', 'Low_DA'}:
    analyte = 'DA'
else:
    analyte = exp

seq  = np.random.permutation(30)[:15]
conc = np.sort(conc_range[exp][seq])
date = datetime.date.today()

wb = xw.Book(cal_path)
sheet = wb.sheets['Calibration']
row = str(sheet.range('A' + str(sheet.cells.last_cell.row)).end('up').row + 1)
sheet.range('A'+row).value = date
sheet.range('B'+row).value = electrode
sheet.range('C'+row).value = analyte
sheet.range('D'+row).value = conc
wb.save()
