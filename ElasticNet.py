'''
Performs the elastic-net regression model to infer
conentrations of neurotransmitters measured by the
fast-scan cyclic voltammetry.
'''

import os
import numpy as np
import pandas as pd
import pickle
import datetime
import glob
import xlwings as xw
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering as AHC
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings; warnings.filterwarnings("ignore")

data_path  = 'Data/EN_data'
fscv_path  = 'Data/FSCV_data'
model_path = 'Data/EN_model'
eval_path  = 'Model_evaluation'
cal_path   = 'calibration_log.xlsx'

## Load calibration log
calibration = pd.read_excel(cal_path)
calibration['Date'] = calibration['Date'].dt.strftime('%Y%m%d')
calibration['Session'] = calibration['Electrode']+'_'+calibration['Date']
calibration['FSCV_data'] = calibration['Date']+'_'+calibration['Electrode']+'_Octaflow'

def check_status(file):
    if os.path.exists(file):
        return True
    else:
        print(f' > File not found: {file.split("/")[-1]}')
        return False

class NT:
    '''
    Prepare data for NT (different neurotransmitters)
    '''

    def __init__(self,target,n_session):
        '''
        INITIATE
            1. self._session:   dict, the sessions associated with that analyte
            2. self.target:     str,  the target session name
            3. self.analyte:    str,  the analyte of interest, should be DA or 5-HT
            4. self._cluster(): fxn,  performs clustering to choose sessions
        '''

        self.target    = target
        self.n_session = n_session
        self._session  = calibration.Session.to_dict()
        self.analyte   = calibration[calibration.Session==self.target].Analyte.values[0]
        self.session   = list(calibration[calibration.Analyte==self.analyte].Session)
        self._cluster()

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self,target):
        if target in calibration.Session.values:
            self._target = target
        else:
            raise ValueError('Wrong input. No such session.')

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self,session):
        if type(session) == str and session in calibration.Session.values:
            self._session = [session]
        elif type(session) == list:
            self._session = [i for i in session if i in calibration.Session.values]
        else:
            raise ValueError('Wrong input. No such session.')

    @property
    def analyte(self):
        return self._analyte

    @analyte.setter
    def analyte(self,analyte):
        self._analyte = analyte


    def prepare(self,var):
        '''
        Prepare the x and y parameters for clustering or elastic net modeling.

        INPUT
            1. var: string of x or y or BL (baseline). The variable of interest to prepare for the data.
            2. diff: boolean default True. Indicate whether the resampled x needs to differentiate.
               Usually used for visualization in the self._cluster() function.
        '''
        if var not in ['x','y','BL']:
            raise ValueError('Input should be x, y, or BL (baseline).')
        if var == 'x' or var == 'BL':
            result = np.empty((0,1000))
            suffix = 'FSCV'
        else:
            result = np.empty((0,))
            suffix = 'CONC'

        un_preprocess = []
        for i in self.session:
            file = os.path.join(data_path,f'{i}_{suffix}.npy')
            if check_status(file):
                fscv = np.load(file)
                if var != 'BL':
                    result = np.concatenate((result,fscv))
                else:
                    result = np.concatenate((result,np.mean(fscv[:400],axis=0)[np.newaxis,:]))
            else:
                un_preprocess.append(i)
        if len(un_preprocess):
            [self.session.remove(i) for i in un_preprocess]

        if var == 'x':
            result = np.diff(result) * 100000 # take finite difference, 100 kHz
            self.x = result
        elif var == 'y':
            self.y = result
        else:
            self.x_BL = result

    def _cluster(self):
        '''
        Perform clustering for electrode selection.
        '''

        def optimal_cluster(data):
            '''
            Return the indeces for the sessions in the cluster.

            INPUT
                1. data: Array of (n,999), where n is the number of sessions for clustering.

            OUPUT
                1. index: Array of (m,), where m is the amount of sessions. m >= n_sessions.
            '''

            n_cluster = data.shape[0]
            if n_cluster <= self.n_session:
                return np.arange(n_cluster)
            while n_cluster > 0:
                y = AHC(n_clusters=n_cluster, linkage='ward').fit_predict(data)
                if len(np.where(y==y[target_session])[0]) < self.n_session:
                    n_cluster -= 1
                else:
                    return np.where(y==y[target_session])[0]

        def link_color(data,index):
            '''
            Sessions that group with our target will return red, otherwise blue.
            '''
            Z = linkage(data,'ward')
            leaf_colors = {i:('r' if i in index else 'b') for i in range(len(Z)+1)}
            link_colors = {}
            for i, ind_12 in enumerate(Z[:,:2].astype(int)):
                c1, c2 = (link_colors[ind] if ind > len(Z) else leaf_colors[ind] for ind in ind_12)
                link_colors[i+1+len(Z)] = c1 if c1 == c2 else 'b'

            return link_colors

        # Prepare all data at once before use of the function resample()
        self.prepare('x')
        self.prepare('y')
        self.prepare('BL')
        x_diff = np.diff(self.x_BL) * 100000 # take finite difference, 100 kHz

        target_session = self.session.index(self.target)
        index = optimal_cluster(x_diff)
        link = link_color(x_diff,index)

        # Reset self.session after clustering
        self.session = [self.session[i] for i in index]

        plt.figure(figsize=(5,8))
        plt.subplot(211)
        dendrogram( linkage(x_diff,'ward'), link_color_func=lambda k:link[k])
        plt.xticks(rotation=90)
        plt.ylabel('Distance')
        plt.xlabel('Session ID')
        plt.title('Clustering of various sessions')
        plt.subplot(212)
        plt.xlabel('Time points (sample)')
        plt.ylabel('Amplitude (nA)')
        plt.title('Representative CV response in each session')
        for i in range(len(x_diff)):
            if i in index:
                plt.plot(self.x_BL[i],'r',alpha=0.2)
            else:
                plt.plot(self.x_BL[i],'b',alpha=0.2)
        plt.subplots_adjust(hspace=0.3)
        plt.savefig(os.path.join(eval_path,f'{self.target}-Cluster-{self.__class__.__name__}.png'))

    def distribution(self):
        '''
        Plot the distribution of data, both before and after resampling.
        '''

        def plot_distribution(data):
            a,b = np.unique(data,return_counts=True)
            plt.bar(a.astype(int).astype(str),b)
            for i,v in enumerate(b):
                plt.text(i, v+5, str(int(v)), ha='center', fontdict=dict(fontsize=8))
            plt.ylabel('Count')
            plt.xlabel('Concentration (nM)')
            plt.xticks(rotation=45)

        if self.analyte != 'pH':
            plt.figure(figsize=(16,4))
            plot_distribution(self.y)
            plt.title('Data distribution before resampling')
            plt.savefig(os.path.join(eval_path,f'{self.target}-Dist-before.png'))
            plt.figure(figsize=(6,4))
            plot_distribution(self.y_resample)
            plt.title('Data distribution after resampling')
            plt.savefig(os.path.join(eval_path,f'{self.target}-Dist-after.png'))

    def resample(self,conc,sigma,size,base):
        '''
        Resample data for EN regression.
        Detailed description in Kishida 2016 PNAS.

        INPUT (All positive integers)
            1. conc: the centered concentration.
            2. sigma: the variance of the sampling.
            3. size: the intended amount of data.
            4. base: the round-up base, default 50 for high DA, 5-HT and NE, 5 for low DA.
        '''

        self.conc  = conc
        self.sigma = sigma
        self.size  = size
        self.model_name = f'{self.analyte}_{self.conc}'

        # Normal about the target concentration, and round to the base
        # (in the high DA situation, 50 nM).
        s = np.random.normal(self.conc, self.sigma, self.size)
        myround = lambda x: base*round(x/base)
        round_s = [myround(i) for i in s]
        sub_conc, counts = np.unique(round_s, return_counts=True)

        # Concatenating samples from each sub_conc
        xx = np.empty((0,999))
        yy = np.empty((0))
        for i in range(len(sub_conc)):
            if sum(self.y==sub_conc[i]) != 0:
                sub_x = self.x[ self.y==sub_conc[i] , : ]
                sub_y = self.y[ self.y==sub_conc[i] ]
                if sum(self.y==sub_conc[i]) > counts[i]:
                    np.random.seed(1)
                    ids = np.random.choice(len(sub_x),counts[i],replace=False)
                    xx  = np.concatenate((xx,sub_x[ids,:]))
                    yy  = np.append(yy,sub_y[ids])
                else:
                    xx  = np.concatenate((xx,sub_x))
                    yy  = np.append(yy,sub_y)

        self.x_resample = xx
        self.y_resample = yy

        # Add 0.1 times amount of data points from PBS
        sub_x = self.x[self.y==0,:]
        sub_y = self.y[self.y==0]
        ids = np.random.choice(len(sub_x),int(self.size/10),replace=False)
        self.x_resample = np.append(self.x_resample,sub_x[ids,:],axis=0)
        self.y_resample = np.append(self.y_resample,sub_y[ids])


class pH(NT):
    '''
    Prepare data for different pH.
    '''

    def __init__(self, target,n_session):
        '''
        INITIATE
            1. self._session:   dict, the sessions associated with that analyte
            2. self.target:     str,  the target session name
            3. self.analyte:    str,  the analyte of interest, should be DA or 5-HT
            4. self._cluster(): fxn,  performs clustering to choose sessions
        '''
        self._session = calibration.Session.to_dict()
        self.analyte  = 'pH'
        self.n_session = n_session

        # Add target into the sessions of pH
        self.target   = target
        self._session = list(calibration[calibration.Analyte==self.analyte].Session)
        self._session.append(self.target)
        self._cluster()

    def resample(self,size):
        '''
        Prepare samples for the pH
        '''
        self.size = size
        xx = np.empty((0,999))
        if sum(self.y==0) > size:
            np.random.seed(1)
            ids = np.random.choice(len(self.x[self.y==0]),size,replace=False)
            xx  = np.concatenate((xx,self.x[ids,:]))
        else:
            xx  = np.concatenate((xx,self.x[self.y==0]))

        yy = np.zeros((xx.shape[0],)) # Always 0 for different pH
        self.x_resample = xx
        self.y_resample = yy


def regression(*data):
    '''
    Build EN regression model with given data.
    '''

    xx = np.empty((0,999))
    yy = np.empty((0,))

    for datum in data:
        if not isinstance(datum,NT):
            raise ValueError('Wrong input. Data should be instances of ElasticNet.')
        elif 'x_resample' not in dir(datum):
            raise AttributeError('Data should be resampled before regression')
        else:
            xx = np.concatenate((xx, datum.x_resample))
            yy = np.concatenate((yy, datum.y_resample))

    if 'model_name' not in dir(data[0]):
        raise AttributeError('The first input should be the instance of NT.')

    xx, yy = shuffle(xx, yy, random_state=0)
    x_train,x_test,y_train,y_test = train_test_split(xx, yy, test_size=0.2)
    l1_ratios = np.arange(1,0,-0.1)
    cv = KFold(n_splits=10, shuffle=True)
    model = ElasticNetCV(l1_ratio=l1_ratios, cv=cv, n_jobs=-1)
    model.fit(x_train, y_train)
    print(f'Model performance: {model.score(x_test,y_test)}')

    # Save all data
    subfolder = os.path.join(model_path,data[0].target)
    if not os.path.exists(subfolder):
        os.mkdir(subfolder)
    pickle.dump(model,open(os.path.join(subfolder,f'{data[0].model_name}.sav'),'wb'))
    np.save(os.path.join(subfolder,f'{data[0].model_name}-xtest.npy'), x_test)
    np.save(os.path.join(subfolder,f'{data[0].model_name}-ytest.npy'), y_test)
    np.save(os.path.join(subfolder,f'{data[0].model_name}-xtrain.npy'), x_train)
    np.save(os.path.join(subfolder,f'{data[0].model_name}-ytrain.npy'), y_train)

    with open(os.path.join(subfolder,'readme.txt'),'a') as f:
        f.write('======================')
        f.write(f'\nModel name: {data[0].model_name}')
        for datum in data:
            f.write('\n----------------------')
            f.write(f'\n > {datum.analyte}: {datum.target}')
            f.write(f'\n > Number of session targeted: {datum.n_session}')
            f.write(f'\n > Session: {datum.session}')
            if datum.analyte == 'pH':
                f.write(f'\n > Resample profile: {datum.size}')
            else:
                f.write(f'\n > Resample profile: {datum.conc}, {datum.sigma}, {datum.size}')
            f.write(f'\n > Shape of x_resample and y_resample: {datum.x_resample.shape}, {datum.y_resample.shape}')
            f.write('\n\n\n\n')
