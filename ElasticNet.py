'''
Performs the elastic-net regression model to infer
conentrations of neurotransmitters measured by the
fast-scan cyclic voltammetry.


TODO:

Docstring
plot_distribution()
save files consideration
README

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

def rand_conc(file,write_in=True):

    sheets = pd.read_excel(cal_path,None).keys()
    today = datetime.date.today().strftime('%Y-%m-%d')

    if file not in sheets:
        raise ValueError('Wrong input. No such page.')
    # TODO: Add pH, 5-HT, NE
    conc_range = {
                'High DA': np.arange(1500,-1,-50),
                'Low DA':  np.arange(150,0,-5)
                  }
    seq  = np.random.permutation(30)[:15]
    conc = np.sort(conc_range[file][seq])
    if write_in:
        wb = xw.Book(cal_path)
        sheet = wb.sheets[file]
        row = str(sheet.range('A' + str(sheet.cells.last_cell.row)).end('up').row + 1)
        sheet.range('A'+row).value = today
        sheet.range('B'+row).value = input('Which electrode? ')
        sheet.range('C'+row).value = conc
        wb.save()
    return conc

def check_status(file):
    if os.path.exists(file):
        return True
    else:
        print(f'File not found: {file.split("/")[-1]}')
        return False

class EN:

    def __init__(self,page):
        self._page = page
        self.session_ID = []
        self.session_names = []
        self._cluster()
        self.x_resample = None
        self.y_resample = None
        self.today = datetime.date.today().strftime('%y%m%d')

    @property
    def page(self):
        return self._page

    @page.setter
    def page(self,page):
        sheets = pd.read_excel(cal_path,None).keys()
        if page in sheets:
            self._page = page
        else:
            raise ValueError('Wrong input. No such page.')


    def prepare(self,var,diff=True):
        '''
        Prepare the x and y parameters for clustering or elastic net modeling.
        '''
        if var not in ['x','y']:
            raise ValueError('Input should be x or y.')

        calibration = pd.read_excel(cal_path, self.page)
        calibration['Date'] = calibration['Date'].dt.strftime('%Y%m%d')

        # If it's unassigned, use all sessions
        if len(self.session_ID) == 0:
            self.session_ID = calibration.index.values

        if var == 'x':
            result = np.empty((0,1000))
            suffix = 'FSCV'
        else:
            result = np.empty((0,))
            suffix = 'CONC'

        for i in range(len(self.session_ID)):
            date      = calibration.iloc[self.session_ID[i]].loc['Date']
            electrode = calibration.iloc[self.session_ID[i]].loc['Electrode']
            file = os.path.join(data_path,f'{electrode}_{date}_{suffix}.npy')
            if check_status(file):
                _ = np.load(file)
                result = np.concatenate((result,_))
                self.session_names.append(f'{electrode}_{date}')

        if diff and var == 'x':
            result = np.diff(result) * 100000 # 100 kHz

        if var == 'x':
            if diff:
                self.x = result
            else:
                return result
        else:
            self.y = result

    def _cluster(self):

        def avg(x):
            '''
            Average for every session (across 400 samples)
            '''
            sample, feature = x.shape
            x_mean = np.empty((0,feature))
            for i in range(int(sample/400)):
                xx = np.mean(x[i*400:(i+1)*400],axis=0)[np.newaxis,:]
                x_mean = np.concatenate((x_mean,xx))
            return x_mean

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

        self.prepare('x')
        self.prepare('y')
        x_raw = self.prepare('x',diff=False)

        x_diff = avg(self.x[self.y==0])
        x_raw  = avg(x_raw[self.y==0])

        print('=============')
        print('All sessions:')
        for id,name in zip(self.session_ID,self.session_names):
            print(f'Session {id}: {name}')

        target_session = int(input(' > The session ID of interest: '))
        n_sessions = int(input(' > At least how many sessions in the cluster: '))
        self.session_ID = optimal_cluster(x_diff)
        link = link_color(x_diff,self.session_ID)

        labels = range(len(x_diff))
        self.target = self.session_names[target_session]
        plt.figure(figsize=(6,8))
        plt.subplot(211)
        dendrogram( linkage(x_diff,'ward'), link_color_func=lambda k:link[k])
        plt.ylabel('Distance')
        plt.xlabel('Session ID')
        plt.title('Clustering of various sessions')
        plt.subplot(212)
        plt.xlabel('Time points (sample)')
        plt.ylabel('Amplitude (nA)')
        plt.title('Representative CV response in each session')
        plt.gcf().text(0.02,0.85,f'Session ID' ,fontsize=8, weight='bold')
        for i in labels:
            plt.gcf().text(0.02,0.83-i*0.02,f'{i}: {self.session_names[i]}',fontsize=8)
            if i in self.session_ID:
                plt.plot(x_raw[i],'r')
            else:
                plt.plot(x_raw[i],'b')
        plt.subplots_adjust(left=0.32,hspace=0.3)
        plt.savefig(os.path.join(eval_path,f'{self.target}-Cluster.png'))

    def resample(self,conc,sigma,size,base):

        self.conc  = conc
        self.sigma = sigma
        self.size  = size
        self.base  = base
        self.model_name = f'{self.target}-Model-{self.conc}-{self.sigma}-{self.size}'

        self.prepare('x')
        self.prepare('y')

        # Normal about the target concentration, and round to the base
        # (in the high DA situation, 50 nM).
        s = np.random.normal(self.conc, self.sigma, self.size)
        myround = lambda x: base*round(x/self.base)
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

    def plot_distribution(self,resampled=False):
        # refer to View_models main()
        if self.y_resample is None:
            print('Data should be resampled before plotting.')
            return None
        else:
            if resampled:
                data = self.y_resample
            else:
                data = self.y
                name = f'Dist-{self.target}-{self.today}.png'
                size = (10,8)
        a,b = np.unique(data,return_counts=True)
        plt.figure(figsize=size)
        plt.bar(a.astype(int).astype(str),b)
        plt.ylabel('Count')
        plt.xlabel('Concentration (nM)')
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(eval_path,name))

    def regression(self):
        if self.y_resample is None:
            raise ValueError('Data should be resampled before plotting.')
        a,b = np.unique(self.y_resample,return_counts=True)
        xx, yy = shuffle(self.x_resample, self.y_resample, random_state=0)
        x_train,x_test,y_train,y_test = train_test_split(xx, yy, test_size=0.2)
        l1_ratios = np.arange(1,0,-0.1)
        cv = KFold(n_splits=10, shuffle=True)
        model = ElasticNetCV(l1_ratio=l1_ratios, cv=cv, n_jobs=-1)
        model.fit(x_train, y_train)

        # Save all data
        pickle.dump(model,open(os.path.join(model_path,f'{self.model_name}.sav'),'wb'))
        np.save(os.path.join(model_path,f'{self.model_name}-xtest.npy'), x_test)
        np.save(os.path.join(model_path,f'{self.model_name}-ytest.npy'), y_test)
        np.save(os.path.join(model_path,f'{self.model_name}-xtrain.npy'), x_train)
        np.save(os.path.join(model_path,f'{self.model_name}-ytrain.npy'), y_train)

        if not check_status(pkl_path):
            df = pd.DataFrame(columns=['Target','Conc','Sigma','Size','Date','Dist','Sessions'])
        df = pd.read_pickle(pkl_path)
        df.loc[f'{self.model_name}'] = [self.target,
                                        self.conc,
                                        self.sigma,
                                        self.size,
                                        self.today,
                                        np.array([a,b]),
                                        self.session_names]
        df.to_pickle(pkl_path)
