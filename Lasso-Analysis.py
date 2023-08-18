'''

Make all the models here.
1. DA model predict DA
2. DA model predict HT
3. HT model predict DA
4. HT model predict HT

'''

#%% Import and global variables

def change_dir(directory):
    os.chdir(directory)

import os
import glob
import numpy   as np
import pandas  as pd
import seaborn as sns
import matplotlib.cm     as cm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import Lasso
import warnings; warnings.filterwarnings('ignore')

conc_range = np.arange(0,1501,50)
pH_range   = np.append(0,np.arange(68,79)/10)
name       = {'DA':'dopamine','HT':'serotonin'}

#%% Functions and classes

class NT:

    def __init__(self,analyte):
        self.analyte = analyte
        self.data = load_data_bundle(self.analyte)
        self.sessions = list(self.data.keys())
        self._sessionOI = [] # session of interest
        if self.analyte != 'PH':
            self._concOI = [50,1500] # concentration of interest
        else:
            self._concOI = [50,550]

    @property
    def sessionOI(self):
        return self._sessionOI

    @sessionOI.setter
    def sessionOI(self,interest):
        if type(interest)!=list:
            interest = [interest]
        self._sessionOI = interest

    @sessionOI.deleter
    def sessionOI(self):
        self._sessionOI = []

    @property
    def concOI(self):
        return self._concOI

    @concOI.setter
    def concOI(self,conc):
        if type(conc)!=list or len(conc)!=2:
            raise ValueError('Input a list of 2 concentrations.')
        if conc[0]>=conc[1]:
            raise ValueError('Second conc should be greater than the first.')
        self._concOI = conc

    def make_xy(self,split=True):

        if not self.sessionOI:
            raise ValueError('Should introduce some sessions')

        start_conc,end_conc = self.concOI
        the_range = np.append(0,np.arange(start_conc,end_conc+1,50))
        construct = np.zeros((len(the_range)*len(self.sessionOI),750,999))

        for i in range(len(self.sessionOI)):

            pbs  = self.data[self.sessionOI[i]][0:1]
            nt   = self.data[self.sessionOI[i]][start_conc//50:end_conc//50+1]
            data = np.concatenate((pbs, nt))
            construct[len(the_range)*i:len(the_range)*(i+1)] = data.transpose(0,2,1)

        self.x = construct.reshape((-1,999))

        conc = np.tile(the_range,(1,750,1)).transpose(2,1,0)
        conc = np.tile(conc,(len(self.sessionOI),1,1))
        y = conc.reshape((-1,1))

        zero = np.zeros((y.shape))
        if self.analyte == 'DA':
            self.yDA,self.yHT = y,zero
        elif self.analyte == 'HT':
            self.yDA,self.yHT = zero,y
        elif self.analyte == 'PH':
            self.yDA,self.yHT = zero,zero

        if split: ## Train test split
            x_train,x_test,yDA_train,yDA_test,yHT_train,yHT_test = \
                train_test_split(self.x,self.yDA,self.yHT,test_size=0.4)
            self.x_train = x_train
            self.yDA_train = yDA_train
            self.yHT_train = yHT_train
            self.x_test  = x_test
            self.yDA_test  = yDA_test
            self.yHT_test  = yHT_test


class Model:

    def __init__(self,DA,HT,PH):

        data,label = [DA,HT,PH],['DA','HT','PH']
        remove = [ind for ind in range(len(data)) if not isinstance(data[ind], NT)]
        for ind in remove[::-1]:
            label.pop(ind)
            data.pop(ind)

        x,yDA,yHT = np.empty((0,999)),np.empty((0,1)),np.empty((0,1))
        for datum in data:
            if not hasattr(datum,'x'):
                datum.make_xy()
            x = np.concatenate((x,datum.x_train))
            yDA = np.concatenate((yDA,datum.yDA_train))
            yHT = np.concatenate((yHT,datum.yHT_train))

        for ind in range(len(data)):
            setattr(self,label[ind],data[ind])

        self.x_train = x
        self.yDA_train = yDA
        self.yHT_train = yHT


    def train(self,alpha,based_on):

        self.alpha = alpha
        self.based_on = based_on
        self.model = Lasso(self.alpha)
        self.model.fit(self.x_train,getattr(self,f'y{based_on}_train'))
        print('\nFinish training')


def check_trials(session):

    trial_header = os.path.join('Data','FSCV_data','Calibrate-50-1500',session,session)
    trials = [trial_header[:-3]+'_PBS']

    analyte = session.split('_')[-1]
    if analyte == 'PH':
        for i in range(1,len(pH_range)): # Nomenclature of our sessions, pH 6.8-7.8
            trials.append(trial_header+f'_{pH_range[i]}')
    else:
        for i in range(1,len(conc_range)):
            trials.append(trial_header+f'_{conc_range[i]}')
    trials = [i+'.hdcv Color.txt' for i in trials]

    if bool([os.path.exists(trial) for trial in trials]):
        return trials
    else:
        raise ValueError('Some files do not exist.')


def data_bundle(session):
    filename = os.path.join('Data','FSCV_data','Calibrate-50-1500','Data_bundle',session+'.npy')
    if not os.path.exists(filename):
        trials = check_trials(session)
        data = np.zeros((len(trials),1000,750))

        for i in range(len(trials)):
            data[i] = np.loadtxt(trials[i])[:,-750:]

        np.save(filename,data)
    else:
        data = np.load(filename)
    return data


def load_data_bundle(analyte):

    change_dir(os.path.join('Data','FSCV_data','Calibrate-50-1500'))
    sessions = glob.glob(f'*{analyte}')
    change_dir()
    raw = {session:data_bundle(session) for session in sessions}

    der = {}
    for key,value in raw.items():
        der[key] = np.diff(value,axis=1) * 100

    return der


def colorbar():
    sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=0, vmax=1500))
    plt.colorbar(sm,label='Concentration (nM)',ticks=np.arange(0,1501,300))


def show_performance(model,inputs):

    '''
    If inputs are strings, then show performance of held-out data of that analyte.
    Otherwise, it has to be a NT instance, the unseen session to be evaluated.
    '''

    def helper():
        colors = np.append(0,np.arange(data.concOI[0],data.concOI[1]+1,50))
        palettes = [cm.rainbow(i//50/31) for i in colors]

        df = pd.DataFrame({'True':true.flatten(),'Pred':pred})
        df['True'] = df['True'].astype('int')
        df[df<0] = 0 # Enforce 0 if predicts negative

        if np.diff(data.concOI)[0]>1000:
            figsize,pointsize = 10,300
        else:
            figsize,pointsize = 6,700

        plt.figure(figsize=(figsize,4))
        sns.stripplot(data=df,x='True',y='Pred',size=1,jitter=0.15,palette=palettes)
        mean  = pd.Series([df[df['True']==i*50]['Pred'].mean() for i in range(31)]).dropna()
        truth = colors if data.analyte == model.based_on else [0]*len(mean)
        plt.scatter(range(len(mean)),mean,c='brown',marker='_',s=pointsize,label='Prediction')
        plt.scatter(range(len(mean)),truth,c='k',marker='_',s=pointsize,label='Truth')
        plt.xlabel(f'True {xaxis} concentration (nM)')
        plt.ylabel(f'Predicted {yaxis} concentration (nM)')
        plt.xticks(rotation=90)
        plt.legend(loc='upper left',frameon=False)
        plt.title(title)

        return df

    if isinstance(inputs,NT): # Unseen data
        data = inputs
        true = getattr(data,f'y{data.analyte}')
        pred = model.model.predict(data.x)
        xaxis = name[data.analyte]
        yaxis = name[model.based_on]
        title = f'Predict unseen {data.analyte} sessions'
        return helper()
    else: # Held out data
        if hasattr(model,inputs):
            data = getattr(model,inputs)
            true = getattr(data, f'y{data.analyte}_test')
            pred = model.model.predict(data.x_test)
            xaxis = name[inputs]
            yaxis = name[model.based_on]
            title = f'Predict held-out {data.analyte} data'
            return helper()
        else:
            print('Model does not contain data of this analyte')

#%% Initiate data

DA = NT('DA')
HT = NT('HT')
PH = NT('PH')

#%% Set up data

DA.sessionOI = [
    '20220616_E103_DA', # 0
    '20220818_E108_DA', # 1
    '20220430_E86_DA',  # 2
    '20220901_E112_DA', # 3
    '20220509_E78_DA',  # 4
    '20220425_E78_DA',  # 5
    '20220523_E84_DA',  # 6
    '20220615_E102_DA', # 7
    '20220623_E102_DA', # 8
    '20220530_E98_DA'   # 9
    ]

HT.sessionOI = [
    '20220613_E90_HT',  # 0
    '20220524_E81_HT',  # 1
    '20220713_E102_HT', # 2
    '20220821_E108_HT', # 3
    '20220607_E78_HT',  # 4
    '20220706_E104_HT', # 5
    '20220520_E78_HT',  # 6
    ]

PH.sessionOI = [
    '20220707_E105_PH', # 0
    '20220706_E104_PH', # 1
    '20220707_E100_PH', # 2
    '20220630_E104_PH', # 3
    '20220629_E100_PH', # 4
    '20220708_E102_PH', # 5
    '20220628_E105_PH', # 6
    '20220825_E110_PH'  # 7
    ]

DA.make_xy()
HT.make_xy()
PH.make_xy()

model = Model(DA,HT,PH)
model.train(0.1,'DA')
