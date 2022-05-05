'''

Predict DA concentrations by a given model

'''

import os
os.chdir('/Users/hungyunlu/Library/CloudStorage/Box-Box/Hung-Yun Lu Research File/Projects/FSCV/EN_FSCV')

import numpy as np
import matplotlib.pyplot as plt
import utils
import pickle
import ElasticNet as EN
from sklearn.linear_model import LinearRegression

EN_data    = r'Data/EN_data'   # EN_data folder path

def load_calibration():
    return [np.loadtxt(i) for i in utils.all_trials(session)]

def load_model():
    return pickle.load(open(os.path.join(EN_data,f'{model_name}.sav'),'rb'))

def plot_model_coefficient(files,model,scale):
    
    frame = files[0][:,500]
    plt.plot(np.diff(frame)*100,c='gray',alpha=0.4)
    plt.scatter(np.arange(999),np.diff(frame)*100,c=model.coef_,
                cmap='bwr',s=0.6,vmin=-scale,vmax=scale)

    plt.xlabel('Time (ms)')
    plt.xticks(np.linspace(0,len(frame),6),np.linspace(0,10,6,dtype=int))
    plt.ylabel('Current change rate (nA/ms)')
    plt.title(f'Model coefficients of [{model_name}]\ncolor coded on session [{session}]')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.bwr, 
                               norm=plt.Normalize(vmin=-scale,vmax=scale))
    plt.colorbar(sm,label='Coefficient')

def analyze_calibration(model,n_sample,low_bound,high_bound):
    x_test,y_test = EN.prepare_samples(session,n_sample,low_bound,high_bound)
    
    y_true,y_pred = np.empty(0,),np.empty(0,)
    for i in np.unique(y_test):
        y_true = np.concatenate((y_true, y_test[y_test==i]))
        y_pred = np.concatenate((y_pred, model.predict(x_test[y_test==i])))
    
    real  = np.arange(low_bound-20,high_bound+20,5)
    line  = LinearRegression().fit(y_true.reshape(-1,1),y_pred)
    pred  = line.coef_*real+line.intercept_
    score = line.score(y_true.reshape(-1,1),y_pred)
    
    plt.scatter(y_true,y_pred,s=0.5,c='k')
    plt.plot(real,real,c='r',label='Identity')
    plt.plot(real,pred,c='k',label=f'$R^2={score:.3f}$')
    plt.xlabel('True concentration (nM)')
    plt.ylabel('Predicted concentration (nM)')
    plt.title(f'Model [{model_name}] predicts session [{session}]')
    plt.legend(loc='upper left',frameon=False)

#%%

session = '20220502_E87'
model_name = 'M05022244'
files = load_calibration()
model = load_model()

plot_model_coefficient(files, model, 50)
plt.show()

analyze_calibration(model, 4000, 0, 500)
plt.show()
