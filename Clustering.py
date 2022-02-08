#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Clustering.py

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering as AHC
from scipy.cluster.hierarchy import linkage, dendrogram
import utils
import matplotlib.pyplot as plt

def avg(x):
    sample, feature = x.shape
    x_mean = np.empty((0,feature))
    for i in range(int(sample/400)):
        xx = np.mean(x[i*400:(i+1)*400],axis=0)[np.newaxis,:]
        x_mean = np.concatenate((x_mean,xx))
    return x_mean


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
    target_session = int(input(' > The session ID of interest: '))
    n_sessions = int(input(' > At least how many sessions in the cluster: '))
    n_cluster = data.shape[0]
    while n_cluster > 0:
        y = AHC(n_clusters=n_cluster, linkage='ward').fit_predict(data)
        if len(np.where(y==y[target_session])[0]) < n_sessions:
            n_cluster -= 1
        else:
            index = np.where(y==y[target_session])[0]
            break
    return index


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


def cluster():

    print(' > Now running cluster() function in Clustering.py')
    page = utils.ask_page()
    x_raw,  sessions = utils.prepare('x',page,diff=False) # For visualization purpose
    x_diff, sessions = utils.prepare('x',page,diff=True,view_session=False)
    y,      sessions = utils.prepare('y',page,view_session=False)

    x_diff = avg(x_diff[y==0])
    x_raw  = avg(x_raw[y==0])

    ind  = optimal_cluster(x_diff)
    link = link_color(x_diff,ind)

    labels = range(len(x_diff))
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
        plt.gcf().text(0.02,0.83-i*0.02,f'{i}: {sessions[i]}',fontsize=8)
        if i in ind:
            plt.plot(x_raw[i],'r')
        else:
            plt.plot(x_raw[i],'b')
    plt.subplots_adjust(left=0.32,hspace=0.3)
    # plt.savefig(os.path.join(utils.eval_path,'Cluster.png'))
    plt.show()

    return ind

if __name__ == '__main__':
    index = cluster()
