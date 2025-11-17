#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:47:13 2023

@author: kbnewhart@mines.edu
"""
import pandas as pd
from os.path import join
import pickle
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

from mrmr import mrmr_regression # pip install mrmr_selection

df = pd.DataFrame(columns=['i', 'RMSE'])

for i in range(15,3,-1):
    print(i)
    
    trainDF, x_train, x_test, y_train, y_test = train_test_cdf_split(hotDeckDF, 
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
    
    trainDF, x_train, x_test, y_train, y_test = train_test_cdf_split(hotDeckDF, 
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
    
    
# 11 is the best! Let's just do a side-by-side
df = pd.DataFrame(columns=['i', 'RMSE'])
for i in [22,11]:
    print(i)
    
    trainDF, x_train, x_test, y_train, y_test = train_test_cdf_split(hotDeckDF, 
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
    
    trainDF, x_train, x_test, y_train, y_test = train_test_cdf_split(hotDeckDF, 
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
    
    
#TODO:  Save the 11 best features and add to the 'project_setup' file
# ['Lagged Pre Disinfection E. coli',
 # 'NSEC NH3 Pre-NH3 Dosing',
 # 'NSEC Inf Temp',
 # 'Upstream Analyzer 1',
 # 'Nitrification_Effluent_NH3_FC24',
 # 'Upstream Analyzer 2',
 # 'NSEC MLSS',
 # 'Nitrification_Effluent_TIN_FC24',
 # 'Nitrification_Effluent_OP_FC24',
 # 'Nitrification_Effluent_TN_FC24',
 # 'North Outfall NO2']