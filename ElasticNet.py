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
eval_path  = 'Data/Model_evaluation'
cal_path   = 'Log/calibration_log.xlsx'
pkl_path   = 'Log/EN.pkl'

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

    def __init__(self,target):
        '''
        INITIATE
            1. self._session:   dict, the sessions associated with that analyte
            2. self.target:     str,  the target session name
            3. self.analyte:    str,  the analyte of interest, should be DA or 5-HT
            4. self._cluster(): fxn,  performs clustering to choose sessions
        '''

        self._session = calibration.Session.to_dict()
        self.target   = target
        self.analyte  = calibration[calibration.Session==self.target].Analyte.values[0]
        self._session = list(calibration[calibration.Analyte==self.analyte].Session)
        self._cluster()

    @property
    def analyte(self):
        return self._analyte

    @analyte.setter
    def analyte(self,analyte):
            self._analyte = analyte

    @property
    def session(self):
        return self._session

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self,target):
        if target in self.session.values():
            self._target = target
        else:
            raise ValueError('Wrong input. No such session.')

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
                2. target_session: Positive integer, the session ID of interest.
                3. n_sessions: Positive integer, the amount of sessions in the cluster.

            OUPUT
                1. index: Array of (m,), where m is the amount of sessions. m >= n_sessions.
            '''

            n_cluster = data.shape[0]
            if n_cluster <= n_sessions:
                return np.arange(n_cluster)
            while n_cluster > 0:
                y = AHC(n_clusters=n_cluster, linkage='ward').fit_predict(data)
                if len(np.where(y==y[target_session])[0]) < n_sessions:
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
        n_sessions = int(input(' > At least how many sessions in the cluster: '))
        index = optimal_cluster(x_diff)
        link = link_color(x_diff,index)

        # Reset self.session after clustering
        self._session = [self.session[i] for i in index]

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
                plt.plot(self.x_BL[i],'r')
            else:
                plt.plot(self.x_BL[i],'b')
        plt.subplots_adjust(hspace=0.3)
        plt.savefig(os.path.join(eval_path,f'{self.target}-Cluster-{self.__class__}.png'))

    def distribution():
        '''
        Unfinished
        '''
        pass

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

        self.conc = conc
        self.model_name = f'{self.analyte}_{self.conc}'

        self.prepare('x')
        self.prepare('y')

        # Normal about the target concentration, and round to the base
        # (in the high DA situation, 50 nM).
        s = np.random.normal(self.conc, sigma, size)
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
        self.x_resample = np.append(self.x_resample,self.x[self.y==0],axis=0)
        self.y_resample = np.append(self.y_resample,self.y[self.y==0])


class pH(NT):
    '''
    Prepare data for different pH.
    '''

    def __init__(self, target):
        '''
        INITIATE
            1. self._session:   dict, the sessions associated with that analyte
            2. self.target:     str,  the target session name
            3. self.analyte:    str,  the analyte of interest, should be DA or 5-HT
            4. self._cluster(): fxn,  performs clustering to choose sessions
        '''
        self._session = calibration.Session.to_dict()
        self.analyte  = 'pH'

        # Add target into the sessions of pH
        self.target   = target
        self._session = list(calibration[calibration.Analyte==self.analyte].Session)
        self._session.append(self.target)
        self._cluster()

    def resample(self,size):
        '''
        Prepare samples for the pH
        '''
        self.prepare('x')
        self.prepare('y')

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
    Build EN regression model with given data
    '''

    xx = np.empty((0,999))
    yy = np.empty((0,))

    for datum in data:
        if not isinstance(datum,ENet.NT):
            raise ValueError('Wrong input. Data should be instances of ElasticNet.')
        else:
            xx = np.concatenate((xx, datum.x_resample))
            yy = np.concatenate((yy, datum.y_resample))

    xx, yy = shuffle(xx, yy, random_state=0)
    x_train,x_test,y_train,y_test = train_test_split(xx, yy, test_size=0.2)
    l1_ratios = np.arange(1,0,-0.1)
    cv = KFold(n_splits=10, shuffle=True)
    model = ElasticNetCV(l1_ratio=l1_ratios, cv=cv, n_jobs=-1)
    model.fit(x_train, y_train)
    
    # Save all data
    # pickle.dump(model,open(os.path.join(model_path,f'{self.model_name}.sav'),'wb'))
    # np.save(os.path.join(model_path,f'{self.model_name}-xtest.npy'), x_test)
    # np.save(os.path.join(model_path,f'{self.model_name}-ytest.npy'), y_test)
    # np.save(os.path.join(model_path,f'{self.model_name}-xtrain.npy'), x_train)
    # np.save(os.path.join(model_path,f'{self.model_name}-ytrain.npy'), y_train)
    #
    # if not check_status(pkl_path):
    #     df = pd.DataFrame(columns=['Target','Conc','Sigma','Size','Date','Dist','Sessions'])
    # df = pd.read_pickle(pkl_path)
    # df.loc[f'{self.model_name}'] = [self.target,
    #                                 self.conc,
    #                                 self.sigma,
    #                                 self.size,
    #                                 self.today,
    #                                 np.array([a,b]),
    #                                 self.session_names]
    # df.to_pickle(pkl_path)
