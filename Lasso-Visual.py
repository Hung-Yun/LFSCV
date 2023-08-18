'''

Make plots for visualization

'''

#%% Import and global variables

import numpy   as np
import pandas  as pd
import matplotlib.cm     as cm
import matplotlib.pyplot as plt
import statsmodels.api   as sm
from sklearn.metrics         import mean_squared_error
import warnings; warnings.filterwarnings('ignore')

np1 = np.load('DA-DA.npy')
np2 = np.load('DA-HT.npy')
np3 = np.load('HT-DA.npy')
np4 = np.load('HT-HT.npy')

name = {'DA':'dopamine','HT':'serotonin'}


#%%

def plot(nps,model,test):


    feat = {'capsize':4,'capthick':2,'ls':'none',
            'marker':'o','markersize':5,'markeredgecolor':'k'}

    df = pd.DataFrame(nps,columns=['True','Pred'])
    df['True'] = df['True'].astype('int')
    mean = pd.Series([df[df['True']==i*50]['Pred'].mean() for i in range(31)]).dropna()
    std  = pd.Series([df[df['True']==i*50]['Pred'].std() for i in range(31)]).dropna()
    reg  = sm.OLS(nps[:,1],nps[:,0]).fit()
    colors = [cm.rainbow(i/31) for i in range(31)]
    for pos, y, err, color in zip(np.arange(31),mean,std,colors):
        plt.errorbar(pos, y, err, color=color,**feat)
    if model==test:
        plt.plot([0,31],[0,1501],c='brown',label='y=x')
        plt.yticks(range(0,1501,250))
    else:
        plt.plot([0,31],[0,0],c='brown',label='y=0')
        plt.ylim([-50,800])
        plt.yticks(range(0,751,250))

    plt.plot([0,31],[0,reg.params[0]*1500],c='grey',label='Prediction')
    plt.xticks(range(0,31,5),range(0,1501,250))
    plt.legend(loc='upper left',frameon=False)
    plt.xlabel(f'True {name[model]} concentration (nM)')
    plt.ylabel(f'Predicted {name[test]} concentration (nM)')

plt.figure(figsize=(10,10))
plt.subplot(221)
plot(np1,'DA','DA')
plt.subplot(222)
plot(np2,'DA','HT')
plt.subplot(223)
plot(np3,'HT','DA')
plt.subplot(224)
plot(np4,'HT','HT')
plt.subplots_adjust(hspace=0.2,wspace=0.3)
plt.show()
