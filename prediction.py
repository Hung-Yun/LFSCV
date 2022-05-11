'''

Predict DA concentrations by a given model

'''

import os
os.chdir('/Users/hungyunlu/Library/CloudStorage/Box-Box/Hung-Yun Lu Research File/Projects/FSCV/EN_FSCV')

import glob
import numpy as np
import matplotlib.pyplot as plt
import utils
import pickle
import ElasticNet as EN
from sklearn.linear_model import LinearRegression

EN_data    = r'Data/EN_data'   # EN_data folder path

def list_models():
    files = glob.glob(os.path.join(EN_data,'*.sav'))
    files.sort(key=lambda x:x.split('.')[0][1:])
    
    return [file.split('/')[-1].split('.')[0] for file in files]


def plot_model_coefficient(files,model,scale):
    
    frame = files[0][:,500]
    plt.plot(np.diff(frame)*100,c='gray',alpha=0.4)
    plt.scatter(np.arange(999),np.diff(frame)*100,c=model.coef_,
                cmap='bwr',s=0.6,vmin=-scale,vmax=scale)

    plt.xlabel('Time (ms)')
    plt.xticks(np.linspace(0,len(frame),6),np.linspace(0,10,6,dtype=int))
    plt.ylabel('Current change rate (nA/ms)')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.bwr, 
                               norm=plt.Normalize(vmin=-scale,vmax=scale))
    plt.colorbar(sm,label='Coefficient')

def plot_calibration_analysis(session,model,n_sample,low_bound,high_bound):
    x_test,y_test = EN.prepare_samples(session,n_sample,low_bound,high_bound)
    
    y_true,y_pred = np.empty(0,),np.empty(0,)
    for i in np.unique(y_test):
        y_true = np.concatenate((y_true, y_test[y_test==i]))
        y_pred = np.concatenate((y_pred, model.predict(x_test[y_test==i])))
    
    # Calculate prediction error
    if np.sum(np.sign(y_pred)) >= 0:
        err1 = np.sum(np.sqrt((y_pred-y_true)**2)) / len(y_pred) # np.mean(low_bound+high_bound)
        model_mean = (high_bound+low_bound)/2
        err2 = np.sum(np.sqrt((y_pred-model_mean)**2)) / len(y_pred)
    else:
        err1,err2 = -np.inf,-np.inf
    
    real  = np.arange(low_bound-20,high_bound+20,5)
    line  = LinearRegression().fit(y_true.reshape(-1,1),y_pred)
    pred  = line.coef_*real+line.intercept_
    score = line.score(y_true.reshape(-1,1),y_pred)
    
    plt.scatter(y_true,y_pred,s=0.5,c='k')
    plt.plot(real,real,c='r',label='Identity')
    plt.plot(real,pred,c='k',label=f'$R^2={score:.3f}$\n$Med={err2:.2f}$\n$True={err1:.2f}$')
    plt.xlabel('True concentration (nM)')
    plt.ylabel('Predicted concentration (nM)')
    plt.legend(loc='upper left',frameon=False)
    
    
    return y_true, y_pred

def main():
    session = '20220502_E87'
    model_name = 'M05022244'
    
    
    files = [np.loadtxt(i) for i in utils.all_trials(session)]
    model = pickle.load(open(os.path.join(EN_data,f'{model_name}.sav'),'rb'))
    
    plot_model_coefficient(files, model, 50)
    plt.title(f'Model coefficients of [{model_name}]\ncolor coded on session [{session}]')
    plt.show()
    
    y_true, y_pred = plot_calibration_analysis(session, model, 4000, 0, 500)
    plt.title(f'Model [{model_name}] predicts session [{session}]')
    plt.show()


if __name__ == '__main__':
    main()
