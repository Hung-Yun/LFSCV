#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## View_models.py
## View all the models that have been created.

import pandas as pd
import os
import utils
import numpy as np
import matplotlib.pyplot as plt

def view(df):
    print('Index\tConc\tSigma\tSize\tDate')
    for i in range(len(df)):
        print(f'{i}\t{df.iloc[i].Conc}\t{df.iloc[i].Sigma}\t{df.iloc[i].Size}\t{df.iloc[i].Date}')

def get_model_name(df,index):
    model_name = df.index[index]
    if os.path.exists(os.path.join(utils.model_path, model_name+'.sav')):
        return model_name
    else:
        return None


def main():
    df = pd.read_pickle('Log/EN.pkl')
    view(df)
    model_name = get_model_name(df, int(input(' > The index of the model: ')) )
    if model_name is not None:
        dist = df.loc[model_name].Distribution
        sessions = df.loc[model_name].Sessions


    index = [str(int(i)) for i in dist[:,0]]
    value = dist[:,1]
    plt.bar(index,value)
    plt.xlabel('Concentration (nM)')
    plt.ylabel('Count')
    plt.title(f'Sample distribution for {model_name}')
    plt.subplots_adjust(left=0.3)
    for i,v in enumerate(value):
        plt.text(i, v+5, str(int(v)), ha='center',fontdict=dict(fontsize=8))

    for i in range(len(sessions)):
        # plt.text(4,3000-i*150, sessions[i],fontdict=dict(fontsize=8))
        plt.gcf().text(0.02,0.9-i*0.04,sessions[i],fontsize=7)
    plt.savefig(os.path.join(utils.eval_path, f'SampleDist-{model_name}.png'))


if __name__ == '__main__':
    main()
