# -*- coding: utf-8 -*-
"""
Created on Mon May 15 19:25:50 2023

@author: kathryn.newhart
"""
from sklearn import model_selection
import keras_tuner as kt # due to this package, we cannot run based on current packages on Macs
from xgboost import XGBRegressor
import numpy as np

def xgb_setup():
    # XG Boost hyperparameter boundaries for exploration by the tuner.
    max_depth = [1,10]
    eta = [0.001,0.5]
    n_estimators = [20,40]
    #constructing the xgb regressor
    def xgb_feat(hp):
      
      xgb_feat = XGBRegressor(
      ## parameters that are tuned with the hyperband tuner:
        # max_depth: max depth of tree, more trees may lead to overfitting, usually 3-10
        max_depth=hp.Int('max_depth', 
                          min_value=max_depth[0], 
                          max_value=max_depth[1], 
                          step=1),
        # eta: learning rate, how much weight each step shrinks, higher eta more conservative, usually 0.01-0.2
        eta=hp.Float('eta', 
                      min_value=eta[0], 
                      max_value=eta[1], 
                      sampling='log'),
        # n_estimators: number of gb trees (eq to boosting rounds)
        n_estimators=hp.Int('n_estimators', 
                            min_value=n_estimators[0],
                            max_value=n_estimators[1], 
                            sampling='linear'),

      # min_child_weight: min sum of weights of all obs required in child, larger is more conservative, default 1
        min_child_weight=1,
        # min_child_weight=hp.Int('min_child_weight',
        # min_value=1,
        # max_value=5,
        # step=1),

      # gamma: minimum loss reduction required to make split at a tree node, large gamma is conservative
        #gamma=0.005,
        gamma=hp.Float('gamma',
        min_value=0.005,
        max_value=0.5),

      # reg_lambda: L2 reg term on weights
        reg_lambda=1,
      # reg alpha: l1 reg term on weights
        reg_alpha=0,

      # colsample_bytree: subsamples ratio of columns (features used) when constructing each tree
        colsample_bytree=0.8,
        # colsample_bytree=hp.Float('colsample_bytree',
        # min_value=0.4,
        # max_value=1),

      # subsample: ratio of  training sample instance, ranges 0 to 1, default 1
        subsample=0.5,
        # subsample=hp.Float('subsample',
        # min_value=0.2,
        # max_value=0.8),

      #seed for column and subsampling
        #random_state = 20,

      # root mean square error is metric for evaluation
        eval_metric='rmse',
      #objective for regression
        objective="reg:squarederror",
        early_stopping_rounds = 30,
      # for feature importance plots
        importance_type = 'gain'
        )
      return xgb_feat


    # k fold random shuffled cross validation (for training set)
    cv = model_selection.KFold(n_splits=10,shuffle=True,random_state=1)

    # making the tuner class
    class CVTuner(kt.engine.tuner.Tuner):
      # function to test model for each trial of hyperparamters tested
      def run_trial(self, trial, x, y):

        rmse = []
        # for each timesplit, build, fit, and test the model
        for train_indices, val_indices in cv.split(x):
          x_train, x_val = x.iloc[train_indices], x.iloc[val_indices]
          y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]
          model = self.hypermodel.build(trial.hyperparameters)
          eval_set = [(x_val, y_val)]
          model.fit(x_train, y_train, eval_set=eval_set,verbose=0)
          rmse.append(model.best_score)
        #update the oracle based on score
        self.oracle.update_trial(trial.trial_id, {'score':np.mean(rmse)})
        #save the best model in each trial, self.save_model did not work
        pickle.dump(model,open(join('data','processed',project,project,'trial_'+trial.trial_id,'model.pkl'),'wb'))
        #self.save_model(trial.trial_id, model)



    # making the hyperband tuner object
    xgb_feat_tuner = CVTuner(
        oracle=kt.oracles.HyperbandOracle(
            objective=kt.Objective('score', 'min'),
            #seed = 1,
            max_epochs=200),
        hypermodel=xgb_feat,
        directory=join('data','processed',project),
        project_name=project,
        overwrite=True
    )
    
    return(xgb_feat_tuner)

