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


def optimal_cluster(data,target_session,n_sessions=5):
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


def plot_cluster(data, sessions, link_colors):
    labels = range(len(data))
    plt.figure(figsize=(6,5))
    d = dendrogram( linkage(data,'ward'), link_color_func=lambda k: link_colors[k])
    plt.ylabel('Distance')
    plt.xlabel('Session ID')
    plt.title('Clustering of various sessions')
    plt.gcf().text(0.02,0.84,f'Session ID' ,fontsize=7.5, weight='bold')
    for i in range(len(sessions)):
        plt.gcf().text(0.02,0.8-i*0.04,f'{i}: {sessions[i]}',fontsize=7.5)
    plt.subplots_adjust(left=0.3)
    plt.savefig(os.path.join(utils.eval_path,'Cluster.png'))


def main():

    # TODO: How to include new session of interest into x_diff and x_raw?

    # x_raw,  sessions = utils.prepare('x',diff=False) # For visualization purpose
    # x_diff, sessions = utils.prepare('x',diff=True)
    y,      sessions = utils.prepare('y')
    x_raw = np.load('x_raw.npy') # Temporary
    x_diff = np.load('x.npy') # Temporary

    x_diff = x_diff[y==0]
    x_raw  = x_raw[y==0]
    x_diff = avg(x_diff)
    x_raw  = avg(x_raw)

    # TODO: change to input
    target_session = 12
    n_session = 6

    ind  = optimal_cluster(x_diff,target_session,n_session)
    link = link_color(x_diff,ind)
    plot_cluster(x_diff, sessions, link)

if __name__ == '__main__':
    main()
