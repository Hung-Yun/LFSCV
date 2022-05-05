'''

Some utility functions

'''

import os
os.chdir('/Users/hungyunlu/Library/CloudStorage/Box-Box/Hung-Yun Lu Research File/Projects/FSCV/EN_FSCV')

import glob
import numpy as np
import matplotlib.pyplot as plt


FSCV_data = r'Data/FSCV_data' # FSCV_data folder path


def check_session(session):
    sessions = glob.glob(os.path.join(FSCV_data,'*'))
    if session not in [sess.split("/")[-1] for sess in sessions]:
        raise ValueError('Session input error')
    else:
        return None

def baseline():
    session_names = [sess.split("/")[-1] for sess in glob.glob(os.path.join(FSCV_data,'*'))]
    
    baseline_name = []
    for session in session_names:
        baseline_name.append(os.path.join(FSCV_data,session,session+'_PBS.hdcv Color.txt'))
        
    baseline_data = {}
    for i in range(len(session_names)):
        baseline_data[session_names[i]] = np.loadtxt(baseline_name[i])[:,500]
        baseline_data[session_names[i]] = np.loadtxt(baseline_name[i])[:,500]
        
    return baseline_data
    

def all_trials(session):
    
    '''
    This is written for high concentration range of DA.
    '''
    
    trial_header = os.path.join(FSCV_data,session,session)
    trials = [trial_header+'_PBS']
    for i in range(1,31):
        trials.append(trial_header+f'_DA_{i*50}') # << High range of DA
    trials = [i+'.hdcv Color.txt' for i in trials]
    
    if bool([os.path.exists(trial) for trial in trials]):
        return trials
    else:
        raise ValueError('Some files do not exist.')

def conc_colorbar(min_conc, max_conc):
    sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=min_conc, vmax=max_conc))
    plt.colorbar(sm,label='Concentration (nM)')