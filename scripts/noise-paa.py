#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 15:09:01 2023

@author: kbnewhart@mines.edu
"""

import pandas as pd
from os.path import join
import pickle
import numpy as np
from mrmr import mrmr_regression # pip install mrmr_selection
project='inf_Ecoli_mrmr'
hotDeckDF = pd.read_pickle(join('data','compiled','hotDeckDF_infEcoli.pkl'))
hotDeckDF['Day'] = hotDeckDF.index.strftime('%y-%m-%d')
hotDeckDF = hotDeckDF.drop_duplicates(subset=['Pre Disinfection E. coli','Day'])
hotDeckDF.drop('Day', axis=1, inplace=True)
y_col = 'Pre Disinfection E. coli'
x_drop=['Analzyer Select', 'Dose Mode Select']
s = 'Standard'
exec(open('scripts/src/cdf_split.py').read()) 
exec(open('/Users/kbnewhart@mines.edu/Documents/GitHub/Newh2oLab/scripts/xgbsetup.py').read())

noise_cost = 0.1
std_devs = hotDeckDF.std()
testDF = hotDeckDF
for column in hotDeckDF.columns:
    std_dev = std_devs[column]
    for index in hotDeckDF.index:
        random_number = np.random.uniform(-noise_cost, noise_cost)
        testDF.at[index, column] += random_number * std_dev

noisyDF = pd.concat([hotDeckDF, testDF], ignore_index=True)

testDF = hotDeckDF
for column in hotDeckDF.columns:
    std_dev = std_devs[column]
    for index in hotDeckDF.index:
        random_number = np.random.uniform(-noise_cost, noise_cost)
        testDF.at[index, column] += random_number * std_dev

noisyDF = pd.concat([noisyDF, testDF], ignore_index=True)

testDF = hotDeckDF
for column in hotDeckDF.columns:
    std_dev = std_devs[column]
    for index in hotDeckDF.index:
        random_number = np.random.uniform(-noise_cost, noise_cost)
        testDF.at[index, column] += random_number * std_dev

noisyDF = pd.concat([noisyDF, testDF], ignore_index=True)




df = pd.DataFrame(columns=['i', 'RMSE'])

for i in range(15,3,-1):
    print(i)
    
    trainDF, x_train, x_test, y_train, y_test = train_test_cdf_split(noisyDF, 
                                                                      y_col, 
                                                                      x_drop,
                                                                      train_frac=0.995,
                                                                      scale=s)
    
    # Select features
    selected_feat = mrmr_regression(X=x_train, y=y_train, K=i)
    
    # Setup xgb model hyperparameter search
    xgb_feat_tuner = xgb_setup()
    
    # Search for best hyperparameters
    xgb_feat_tuner.search(x_train[selected_feat], y_train) 
    
    # Select model with best hyperparameters
    best_feat_hps = xgb_feat_tuner.get_best_hyperparameters(num_trials=1)[0]
    xgb_feat_tuned = xgb_feat_tuner.hypermodel.build(best_feat_hps)
    
    trainDF, x_train, x_test, y_train, y_test = train_test_cdf_split(noisyDF, 
                                                                      y_col, 
                                                                      x_drop,
                                                                      train_frac=0.8,
                                                                      scale=s)
    
    # Validate on all testing data for final model
    eval_set = [(x_test, y_test)]
    tuned_model = xgb_feat_tuned.fit(x_train,y_train,eval_set=eval_set, verbose=False)

    y_pred = tuned_model.predict(x_test)
    rmse = np.mean((y_pred-y_test)**2) ** 0.5
    print(rmse)
    
    row_data = {'i': i, 'RMSE': rmse}  # Modify this line to compute the desired values
    df = df.append(row_data, ignore_index=True)
    
    