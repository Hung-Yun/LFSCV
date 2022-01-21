#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Clustering.py

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering as AHC
from scipy.cluster import hierarchy
import utils

## Folder info
data_path = utils.data_path

## Load calibration log
calibration = utils.load_calibration_log()

## Prepare data
# Cannot use utils.prepare() because we are averaging the PBS responses
xs = np.empty((0,1000))
x_diffs = np.empty((0,999))
for session in range(len(calibration)):
    date      = calibration.iloc[session].loc['Date']
    electrode = calibration.iloc[session].loc['Electrode']
    file = os.path.join(data_path,f'{electrode}_{date}_FSCV.npy')
    if utils.check_status(file):
        x = np.load(file)
        xx = np.mean(x[:400,:],0)[np.newaxis,:] # The average PBS response
        xs = np.concatenate((xs, xx)) # xs for visualization only
        x_diff = np.diff(xx) * 100000
        x_diffs = np.concatenate((x_diffs,x_diff))

## Fit the clustering model
model = AHC(n_clusters=None, distance_threshold=0, linkage='single')
model = model.fit(x_diffs)

## Compute the linkage
counts = np.zeros(model.children_.shape[0])
n_samples = len(model.labels_)
for i, merge in enumerate(model.children_):
    current_count = 0
    for child_idx in merge:
        if child_idx < n_samples:
            current_count += 1  # leaf node
        else:
            current_count += counts[child_idx - n_samples]
    counts[i] = current_count
link = np.column_stack([model.children_, model.distances_, counts])

## Visualization
hierarchy.set_link_color_palette(['red','blue','green'])
R = hierarchy.dendrogram(link,
                         color_threshold = 0.5*max(link[:,2]),
                         above_threshold_color = 'k')
plt.xlabel('Session number')
plt.ylabel('Distance')
plt.title('Hierarchical clustering')
plt.show()

## Visualization
def find_color(R,color):
    leaves  = np.array(R['leaves'])
    indices = np.array([i for i,j in enumerate(R['leaves_color_list']) if j==color])
    return leaves[indices]
red  = find_color(R,'red')
blue = find_color(R,'blue')
green = find_color(R,'green')

plt.xlabel('Data point')
plt.ylabel('Amplitude (nA)')
plt.title('Color-coded FSCV response')
plt.yticks(rotation=90)
for i in range(len(xs)):
    if i in red:
        plt.plot(xs[i,:],c='red')
    elif i in blue:
        plt.plot(xs[i,:],c='blue')
    elif i in green:
        plt.plot(xs[i,:],c='green')
    else:
        plt.plot(xs[i,:],c='black')
plt.show()
hierarchy.set_link_color_palette(None)
