'''

The main script to perform elastic-net regression.

'''

import os
os.chdir('/Users/hungyunlu/Library/CloudStorage/Box-Box/Hung-Yun Lu Research File/Projects/FSCV/EN_FSCV')

import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from utils import all_trials

plt.rcParams["figure.figsize"] = (8,6)
plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams["font.sans-serif"] = 'Tahoma'


FSCV_data = r'Data/FSCV_data' # FSCV_data folder path
EN_data   = r'Data/EN_data'   # EN_data folder path

scan = np.concatenate((np.linspace(-0.6,1.4,500),np.linspace(1.4,-0.6,500)))
samples, low_bound, high_bound = 2000,500,1000

sessions = [
            # '20220420_E57',
            # '20220422_E77',
            # '20220425_E78',
            '20220426_E81_DA',
            '20220427_E82_DA',
            '20220428_E77_DA',
            '20220429_E82',
            '20220430_E86',
            '20220502_E84'
            ]


def prepare_samples(session, n_sample, low_bound, high_bound):
    
    '''
    This is written for high concentration range of DA/HT.
    '''
    
    trials = all_trials(session)
    files  = [np.loadtxt(trial) for trial in trials]
    diffs  = [np.diff(file[:,-500:],axis=0) for file in files]
    
    x = np.empty((999,0))
    for i in diffs:
        x = np.concatenate((x,i),axis=1)
        
    x = x.T
    y = np.repeat(np.arange(0,1501,50),500) # << High range of DA
    
    # Resampling: take samples within the bounds
    x = x[(y>=low_bound)&(y<=high_bound)]
    y = y[(y>=low_bound)&(y<=high_bound)]
    
    # Randomly select n_sample from the range
    index = np.random.randint(0,len(x),n_sample)
    x,y = x[index],y[index]
    
    return x, y


def regression(x, y):
    
    xx, yy = shuffle(x, y)
    x_train,x_test,y_train,y_test = train_test_split(xx, yy, test_size=0.2)

    model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1], 
                         cv=KFold(n_splits=10, shuffle=True), 
                         alphas=np.linspace(0,1,11),
                         n_jobs=2, 
                         max_iter=4000,
                         fit_intercept=True)
    
    model.fit(x_train, y_train)
    score = model.score(x_test,y_test)
    
    y_true,y_pred = np.empty(0,),np.empty(0,)
    
    for i in np.unique(y_test):
        y_true = np.concatenate((y_true, y_test[y_test==i]))
        y_pred = np.concatenate((y_pred, model.predict(x_test[y_test==i])))
    
    return model, y_true, y_pred, score


def fit():
    
    x,y = np.empty((0,999)),np.empty((0,))
    for session in sessions:
        x_,y_ = prepare_samples(session, samples, low_bound, high_bound)
        x = np.concatenate((x,x_))
        y = np.concatenate((y,y_))
    
    model, y_true, y_pred, score = regression(x,y)
    model_name = f'M{datetime.now().strftime("%m%d%H%M")}'   
    pickle.dump(model,open(os.path.join(EN_data,f'{model_name}.sav'),'wb'))
    
    return model, model_name, y_true, y_pred, score


def plot_model(model, model_name, y_true, y_pred, score):
    real = np.arange(low_bound-20,high_bound+20,5)
    line = LinearRegression().fit(y_true.reshape(-1,1),y_pred)
    pred = line.coef_*real+line.intercept_
    
    plt.scatter(y_true,y_pred,s=0.5,c='k')
    plt.plot(real,real,c='r',label='Identity')
    plt.plot(real,pred,c='k',label='Prediction')
    plt.xlim([low_bound-50,high_bound+50])
    plt.ylim([low_bound-300,high_bound+300])
    plt.xlabel('True concentration (nM)')
    plt.ylabel('Predicted concentration (nM)')
    plt.title(f'Model [{model_name}] performance on held-out data\n$R^2={score:.3f}$; $alpha={model.alpha_:.2f}$; $L_1ratio={model.l1_ratio_:.2f}$')
    plt.legend(loc='upper left',frameon=False)
    plt.savefig(os.path.join(EN_data,model_name+'.png'))


if __name__ == '__main__':
    
    model, model_name, y_true, y_pred, score = fit()
    plot_model(model, model_name, y_true, y_pred, score)

    # model = pickle.load(open(os.path.join(EN_data,f'{model_name}.sav'),'rb'))

