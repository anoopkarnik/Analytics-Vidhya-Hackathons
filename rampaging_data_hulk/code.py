import pandas as pd 
import xgboost as xgb
import numpy as np
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
from xgboost.sklearn import XGBClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
ID = 'ID'
target = 'Outcome'
train['source'] = 'train'
test['source'] = 'test'

combi = pd.concat([train,test])

combi.Three_Day_Moving_Average = combi.Three_Day_Moving_Average.fillna(combi.Three_Day_Moving_Average.mean())
combi.Five_Day_Moving_Average = combi.Five_Day_Moving_Average.fillna(combi.Five_Day_Moving_Average.mean())
combi.Ten_Day_Moving_Average = combi.Ten_Day_Moving_Average.fillna(combi.Ten_Day_Moving_Average.mean())
combi.Twenty_Day_Moving_Average = combi.Twenty_Day_Moving_Average.fillna(combi.Twenty_Day_Moving_Average.mean())
combi.Average_True_Range  = combi.Average_True_Range.fillna(combi.Average_True_Range.mean())
combi.Positive_Directional_Movement   = combi.Positive_Directional_Movement.fillna(combi.Positive_Directional_Movement.mean())
combi.Negative_Directional_Movement  = combi.Negative_Directional_Movement.fillna(combi.Negative_Directional_Movement.mean())

train_final = combi[combi['source'] == 'train']
test_final = combi[combi['source'] == 'test']

predictors = [x for x in train_final.columns if x not in [target, ID,'source','Stock_ID']]

xgb = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic', 
 nthread=4,
 scale_pos_weight=1,
 seed=27)

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic', 
 nthread=4,
 scale_pos_weight=1,
 seed=27)

sol2 = pd.DataFrame(columns = ['ID','Outcome'])
for stock in train_final.Stock_ID.unique():
    train_stock = train_final[train_final['Stock_ID'] == stock]
    test_stock = test_final[test_final['Stock_ID'] == stock] 
    xgb.fit(train_stock[predictors], train_stock[target],eval_metric='logloss')
    test_stock[target] = xgb.predict(test_stock[predictors])
    sub = test_stock[['ID','Outcome']]
    sol2 = sol2.append(sub, ignore_index = True)
	
s = pd.DataFrame()
for stock in list(set(a) - set(b)):
    test_stock1 = test_final[test_final['Stock_ID'] == stock]
    s= s.append(test_stock1)
xgb1.fit(train_final[predictors], train_final[target],eval_metric='logloss')
s[target] = xgb1.predict(s[predictors])
sub2 = s[['ID','Outcome']]

sol3 = sol2.append(sub2)
sol3.to_csv('sol5.csv',index =False)