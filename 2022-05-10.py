'''

TEST METRICS TO QUANTIFY ERROR

'''

import os
os.chdir('/Users/hungyunlu/Library/CloudStorage/Box-Box/Hung-Yun Lu Research File/Projects/FSCV/EN_FSCV')

import numpy as np
import matplotlib.pyplot as plt
import utils
import pickle

EN_data    = r'Data/EN_data'   # EN_data folder path

plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams["font.sans-serif"] = 'Tahoma'

import ElasticNet as EN
from sklearn.linear_model import LinearRegression



#%% Old clustering function

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering as AHC

def optimal_cluster(data, n_session,target_session,link_method):
    '''
    Return the indeces for the sessions in the cluster.

    INPUT
        1. data: Array of (n,999), where n is the number of sessions for clustering.

    OUPUT
        1. index: Array of (m,), where m is the amount of sessions. m >= n_sessions.
    '''

    n_cluster = data.shape[0]
    if n_cluster <= n_session:
        return np.arange(n_cluster)
    while n_cluster > 0:
        y = AHC(n_clusters=n_cluster, linkage=link_method).fit_predict(data)
        if len(np.where(y==y[target_session])[0]) < n_session:
            n_cluster -= 1
        else:
            return np.where(y==y[target_session])[0]

def link_color(data,index,link_method):
    '''
    Sessions that group with our target will return red, otherwise blue.
    '''
    Z = linkage(data,link_method)
    leaf_colors = {i:('r' if i in index else 'b') for i in range(len(Z)+1)}
    link_colors = {}
    for i, ind_12 in enumerate(Z[:,:2].astype(int)):
        c1, c2 = (link_colors[ind] if ind > len(Z) else leaf_colors[ind] for ind in ind_12)
        link_colors[i+1+len(Z)] = c1 if c1 == c2 else 'b'

    return link_colors



#%% Analyze baselines
baseline  = utils.baseline()

baseline_key   = list(baseline.keys())
baseline_value = list(baseline.values())

#%% 
target = '20220430_E86'
target_ind = baseline_key.index(target)

for j in range(9):
    plt.plot(np.diff(baseline_value[j])*100,c='gray',alpha=0.4)
plt.plot(np.diff(baseline_value[target_ind])*100,c='r')
plt.xticks(np.linspace(0,998,6),np.linspace(0,10,6))
plt.ylabel('Current change rate (nA/ms)')
plt.xlabel('Time (ms)')
plt.title(f'Baseline response in {target}')


#%%

# link_method = 'average'

x_diff = np.diff(baseline_value) * 100

for link_method in ['ward','complete','average','single']:
    
    index = optimal_cluster(x_diff,2,target_ind,link_method)
    link = link_color(x_diff,index,link_method)
    dendrogram( linkage(x_diff,link_method), link_color_func=lambda k:link[k],
               labels=baseline_key,orientation='right')
    plt.xlabel('Distance')
    plt.ylabel('Session')
    plt.title(f'Distance between sessions using "{link_method}"')
    plt.show()


# for i in range(len(x_diff)):
#     if i in index:
#         plt.plot(x_diff[i],'r')
#     else:
#         plt.plot(x_diff[i],'gray',alpha=0.4)
# plt.xticks(np.linspace(0,998,6),np.linspace(0,10,6))
# plt.ylabel('Current change rate (nA/ms)')
# plt.xlabel('Time (ms)')
# plt.title(f'Clustered sessions with {target}')


#%% 

session = '20220502_E87'
model_names =['M05012118','M05012112','M05012020']
n_sample = 2000

bounds = [(0,500),(500,1000),(1000,1500)]

#%%

from prediction import plot_calibration_analysis

plt.figure(figsize=(18,18))
k=1
for i in range(3):
    model = pickle.load(open(os.path.join(EN_data,f'{model_names[i]}.sav'),'rb'))
    for j in range(3):
        plt.subplot(3,3,k)
        y_true, y_pred = plot_calibration_analysis(session, model, n_sample,*bounds[j])
        plt.title(f'Train {bounds[i]}, test {bounds[j]}')
        k+=1
        
plt.subplots_adjust(wspace=0.3,hspace=0.3)
plt.show()

#%%

true,pred,ks = np.empty(0,),np.empty(0,),[]

for conc in range(0,1501,50):
    
    low_bound,high_bound = conc,conc
    x_test,y_true = EN.prepare_samples(session,1000,low_bound,high_bound)
    
    e1,e2 = [],[]
    
    for i in range(3):
        model = pickle.load(open(os.path.join(EN_data,f'{model_names[i]}.sav'),'rb'))
        y_pred = model.predict(x_test)
        if np.sum(np.sign(y_pred)) >= 0:
            model_mean = np.sum(bounds[i])/2
            # err1 = np.sum(np.sqrt((y_pred-y_true)**2)) / len(y_pred)
            err2 = np.sum(np.sqrt((y_pred-model_mean)**2)) / len(y_pred)
        else:
            err1,err2 = -np.inf,-np.inf
        # e1.append(err1)
        e2.append(err2)
        
    # print('Error_true\t',e1) 
    print('Error_median\t',e2)
    
    k = np.argmin(e2)
    
    model = pickle.load(open(os.path.join(EN_data,f'{model_names[k]}.sav'),'rb'))
    y_pred = model.predict(x_test)
    
    true = np.concatenate((true,np.repeat(conc,1000)))
    pred = np.concatenate((pred,y_pred))
    ks.append(k)
    
line  = LinearRegression().fit(true.reshape(-1,1),pred)
pred_line  = line.coef_*true+line.intercept_
score = line.score(true.reshape(-1,1),pred)

#%%

plt.scatter(true,pred,s=0.5,c='k')
plt.plot(true,true,c='r',label='Identity')
plt.plot(true,pred_line,c='k',label=f'$R^2={score:.3f}$')
plt.xlabel('True concentration (nM)')
plt.ylabel('Predicted concentration (nM)')
plt.xticks(np.arange(0,1501,300))
plt.legend(loc='upper left',frameon=False)
plt.title('Models predict an unseen session\nColor-code the selected models')

for i in range(len(ks)):
    if ks[i] == 0:
        plt.fill_between(range(50*i-10,50*i+10), 2000, color='r',alpha=0.2,ec=None)
    elif ks[i] == 1:
        plt.fill_between(range(50*i-10,50*i+10), 2000, color='k',alpha=0.2,ec=None)
    else:
        plt.fill_between(range(50*i-10,50*i+10), 2000, color='b',alpha=0.2,ec=None)
plt.show()


#%%

