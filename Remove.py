#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Removing unwanted model

import pandas as pd
import os
import utils
import numpy as np
import glob
from View_models import view, get_model_name

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

df = pd.read_pickle('Log/EN.pkl')
logger.warning('Removing unwanted model and associated files from EN_model folder.')
view(df)
model_name = get_model_name(df, int(input(' > The index of the model: ')) )
logger.warning(f'Removing {model_name}.')

if int(input('Sure want to remove (0/1)? ')):
    df = df[df.index != model_name]
    df.to_pickle('Log/EN.pkl')
    print(df)
    files = glob.glob(os.path.join(utils.model_path,f'{model_name}*'))
    for file in files:
        os.remove(file)
    files = glob.glob(os.path.join(utils.eval_path,f'*{model_name}*'))
    for file in files:
        os.remove(file)
