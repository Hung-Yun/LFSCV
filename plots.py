'''

Plots

'''

import os
os.chdir('/Users/hungyunlu/Library/CloudStorage/Box-Box/Hung-Yun Lu Research File/Projects/FSCV/EN_FSCV')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams["figure.figsize"] = (8,6)
plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams["font.sans-serif"] = 'Tahoma'

os.chdir('/Users/hungyunlu/Library/CloudStorage/Box-Box/Hung-Yun Lu Research File/Projects/FSCV/EN_FSCV')
import utils

baseline  = utils.baseline()

FSCV_data = r'Data/FSCV_data' # FSCV_data folder path
scan = np.concatenate((np.linspace(-0.6,1.4,500),np.linspace(1.4,-0.6,500)))


def baseline_NonBG():
    i=0
    for name,fscv in baseline.items():
        plt.plot(scan,fscv,label=name,color=cm.rainbow(i/len(baseline)))
        i += 1
    
    plt.legend(bbox_to_anchor=(1.02,1),
               loc='upper left',
               borderaxespad=0,
               frameon=False,
               fontsize=10)
    plt.xticks([-0.6,-0.1,0.4,0.9,1.4])
    plt.ylabel('Amplitude (nA)')
    plt.xlabel('Voltage (V)')
    plt.title('Baseline response in different sessions')
    plt.show()

def baseline_BG():
    i=0
    for name,fscv in baseline.items():
        plt.plot(np.diff(fscv)*100,label=name,color=cm.rainbow(i/len(baseline)))
        i += 1
    
    plt.legend(bbox_to_anchor=(1.02,1),
               loc='upper left',
               borderaxespad=0,
               frameon=False,
               fontsize=10)
    plt.xticks(np.linspace(0,998,6),np.linspace(0,10,6))
    plt.ylabel('Current change rate (nA/ms)')
    plt.xlabel('Time (ms)')
    plt.title('Baseline response in different sessions, BG')
    plt.show()
    
    
def baseline_analysis(session):
    for name,fscv in baseline.items():
        if name != session:
            plt.plot(np.diff(fscv)*100,c='k',alpha=0.2)
        else:
            plt.plot(np.diff(fscv)*100,c='r')

    plt.xticks(np.linspace(0,998,6),np.linspace(0,10,6))
    plt.ylabel('Current change rate (nA/ms)')
    plt.xlabel('Time (ms)')
    plt.title(f'Session {session}')
    plt.show()

def color_SingleTrial(trial):
    plt.imshow(np.flipud(trial),aspect='auto')
    plt.yticks([0,500,1000],[-0.6,1.4,-0.6])
    plt.xticks(np.arange(0,751,250),np.arange(0,76,25))
    plt.colorbar(label='Amplitude (nA)')
    plt.xlabel('Time (second)')
    plt.ylabel('Voltage (V)')


def scan_NonBG_AllTrials(session, files):
    
    for i,file in enumerate(files):
        plt.plot(scan,file[:,500],color=cm.rainbow(i/len(files)))

    plt.xlabel('Voltage (V)')
    plt.xticks([-0.6,-0.1,0.4,0.9,1.4])
    plt.ylabel('Amplitude (nA)')
    plt.title(f'All trials in {session}')


def scan_BG_AllTrials(session, files):
    
    BG = files[0]
    for i,file in enumerate(files[1:]):
        plt.plot(scan,file[:,500]-BG[:,500],color=cm.rainbow(i/len(files)),alpha=0.6)

    plt.xlabel('Voltage (V)')
    plt.xticks([-0.6,-0.1,0.4,0.9,1.4])
    plt.ylabel('Amplitude (nA)')
    plt.title(f'All trials in {session}, BG')


def derivative_AllTrials(session,files):
    
    for i,file in enumerate(files):
        frame = file[:,500]
        plt.plot(np.diff(frame)*100,color=cm.rainbow(i/len(files)))

    plt.xlabel('Time (ms)')
    plt.xticks(np.linspace(0,len(frame),6),np.linspace(0,10,6,dtype=int))
    plt.ylabel('Current change rate (nA/ms)')
    plt.title(f'All trials in {session}, derivative')


def session_plots(key):
    
    '''
    key is the session name: yyyymmdd_E##_DA/HT
    '''
    
    files  = [np.loadtxt(i) for i in utils.all_trials(key)]
    
    scan_NonBG_AllTrials(key,files)
    utils.conc_colorbar(50,1500)
    plt.show()

    scan_BG_AllTrials(key,files)
    utils.conc_colorbar(50,1500)
    plt.show()

    derivative_AllTrials(key, files)
    utils.conc_colorbar(50,1500)
    plt.show()
    

key = '20220422_E77_DA'

# files  = [np.loadtxt(i) for i in utils.all_trials(key)]

# derivative_AllTrials(key,files)
# utils.conc_colorbar(50,1500)
# plt.show()

session_plots(key)


