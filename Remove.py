#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Remove unnecessary files

import pandas as pd
import os
import utils
import numpy as np

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

model_path = utils.model_path

while True:
    logger.warning('Input model info.')
    mu    = int(input(' > The centered concentration: '))
    sigma = int(input(' > The variance of the distribution: '))
    size  = int(input(' > The amount of samples in total: '))
    base  = int(input(' > Round up base: '))
    date  = int(input(' > The date when the model was created: '))
    model_base = f'Model-{mu}-{sigma}-{size}-{date}'
    if os.path.exists(os.path.join(model_path, model_base+'.sav')):
        break
    else:
        logger.warning('File does not exist. Try again.')
