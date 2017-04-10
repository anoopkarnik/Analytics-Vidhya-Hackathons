import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
import nltk
#from fuzzywuzzy import fuzz
import re
from sklearn.ensemble import GradientBoostingClassifier

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

train['data'] = 'train'
test['data'] = 'test'
test['Surge_Pricing_Type'] = 2
combi = pd.concat([train,test])

combi.Confidence_Life_Style_Index = combi.Confidence_Life_Style_Index.fillna('D')
combi.Customer_Since_Months = combi.Customer_Since_Months.fillna(combi.Customer_Since_Months.median())
combi.Life_Style_Index = combi.Life_Style_Index.fillna(combi.Life_Style_Index.mean())
combi.Type_of_Cab= combi.Type_of_Cab.fillna('F')
combi.Var1 = combi.Var1.fillna(int(combi.Var1.mean()))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_to_encode = ['Confidence_Life_Style_Index','Gender','Destination_Type','Type_of_Cab']
for col in var_to_encode:
    combi[col] = le.fit_transform(combi[col])
combi = pd.get_dummies(combi, columns=var_to_encode)

target = 'Surge_Pricing_Type'
IDcol = 'Trip_ID'
mod_train = combi[combi['data'] == 'train']
mod_test = combi[combi['data'] == 'test']

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgb_param['num_class'] = 4
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'],nfold=cv_folds,
            metrics='merror', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions)

predictors = [x for x in mod_train.columns if x not in [target, IDcol,'data']]
	

param_test1 = {
 'max_depth':range(3,10,1),
 'min_child_weight':range(1,10,1)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=33), 
 param_grid = param_test1, scoring='accuracy',n_jobs=4,iid=False, cv=5)
gsearch1.fit(mod_train[predictors],mod_train[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=gsearch1.best_params_['max_depth'],
 min_child_weight=gsearch1.best_params_['min_child_weight'], gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=33), 
 param_grid = param_test3, scoring='accuracy',n_jobs=4,iid=False, cv=5)
gsearch3.fit(mod_train[predictors],mod_train[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

param_test4 = {
 'subsample':[i/100.0 for i in range(60,100,1)],
 'colsample_bytree':[i/100.0 for i in range(60,100,1)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=gsearch1.best_params_['max_depth'],
 min_child_weight=gsearch1.best_params_['min_child_weight'], gamma=gsearch3.best_params_['gamma'], subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=33), 
 param_grid = param_test4, scoring='accuracy',n_jobs=4,iid=False, cv=5)
gsearch4.fit(mod_train[predictors],mod_train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

param_test6 = {
 'reg_alpha':[0, 0.001, 0.005, ,1e-5, 1e-2, 0.1,0.5, 1,10, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=gsearch1.best_params_['max_depth'],
 min_child_weight=gsearch1.best_params_['min_child_weight'], gamma=gsearch3.best_params_['gamma'], subsample=gsearch4.best_params_['subsample']
												  , colsample_bytree= gsearch4.best_params_['colsample_bytree'],
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1,seed=33), 
 param_grid = param_test6, scoring='accuracy',n_jobs=4,iid=False, cv=5)
gsearch6.fit(mod_train[predictors],mod_train[target])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_


	
xgb2 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=gsearch1.best_params_['max_depth'],
 min_child_weight=gsearch1.best_params_['min_child_weight'],
 gamma=gsearch3.best_params_['gamma'],
 subsample=gsearch4.best_params_['subsample'],
 colsample_bytree=gsearch4.best_params_['colsample_bytree'],
 objective= 'multi:softmax', 
 nthread=4,
 reg_alpha = gsearch6.best_params_['reg_alpha']
 scale_pos_weight=1,
 seed=33)
 


xgb2.fit(mod_train[predictors], mod_train[target],eval_metric='error')
mod_test[target] = xgb1.predict(mod_test[predictors])

sol2 = mod_test[['Trip_ID','Surge_Pricing_Type']]
sol2.to_csv('sol2.csv')