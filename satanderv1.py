# -*- coding: utf-8 -*-
"""
Created on Fri Jul 06 14:41:06 2018

@author: M29480
"""

import numpy as np
import pandas as pd
import warnings

from sklearn import model_selection
from sklearn import ensemble
from scipy.stats import ks_2samp

from sklearn import random_projection
from sklearn.preprocessing import scale
import gc
from copy import deepcopy
import time
import progressbar

from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb

from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from catboost import CatBoostRegressor

notebookstart= time.time()
import seaborn as sns
import matplotlib.pyplot as plt


################################### Combined ensemble using Light and Extreme Gradient Boosting

warnings.filterwarnings("ignore")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ID = test['ID']
y_train = train['target']
y_train = np.log1p(y_train)
train.drop("ID", axis = 1, inplace = True)
train.drop("target", axis = 1, inplace = True)
test.drop("ID", axis = 1, inplace = True)

zero_std_cols = train.columns[train.std() == 0]
train.drop(zero_std_cols, axis=1, inplace=True)
test.drop(zero_std_cols, axis=1, inplace=True)
NUM_OF_DECIMALS = 32
train = train.round(NUM_OF_DECIMALS)
test = test.round(NUM_OF_DECIMALS)
colsToRemove = []
columns = train.columns
for i in range(len(columns)-1):
    v = train[columns[i]].values
    dupCols = []
    for j in range(i + 1,len(columns)):
        if np.array_equal(v, train[columns[j]].values):
            colsToRemove.append(columns[j])
train.drop(colsToRemove, axis=1, inplace=True) 
test.drop(colsToRemove, axis=1, inplace=True) 
train.shape

NUM_OF_FEATURES = 1000
def rmsle(y, pred):
    return np.sqrt(np.mean(np.power(y - pred, 2)))

x1, x2, y1, y2 = model_selection.train_test_split(
    train, y_train.values, test_size=0.20, random_state=5)
model = ensemble.RandomForestRegressor(n_jobs=-1, random_state=7)
model.fit(x1, y1)
print(rmsle(y2, model.predict(x2)))

col = pd.DataFrame({'importance': model.feature_importances_, 'feature': train.columns}).sort_values(
    by=['importance'], ascending=[False])[:NUM_OF_FEATURES]['feature'].values
train = train[col]
test = test[col]
train.shape

THRESHOLD_P_VALUE = 0.005 
THRESHOLD_STATISTIC = 0.25
diff_cols = []
for col in train.columns:
    statistic, pvalue = ks_2samp(train[col].values, test[col].values)
    if pvalue <= THRESHOLD_P_VALUE and np.abs(statistic) > THRESHOLD_STATISTIC:
        diff_cols.append(col)
for col in diff_cols:
    if col in train.columns:
        train.drop(col, axis=1, inplace=True)
        test.drop(col, axis=1, inplace=True)
train.shape

ntrain = len(train)
ntest = len(test)
tmp = pd.concat([train,test])#RandomProjection
weight = ((train != 0).sum()/len(train)).values

tmp_train = train[train!=0]
tmp_test = test[test!=0]
train["weight_count"] = (tmp_train*weight).sum(axis=1)
test["weight_count"] = (tmp_test*weight).sum(axis=1)
train["count_not0"] = (train != 0).sum(axis=1)
test["count_not0"] = (test != 0).sum(axis=1)
train["sum"] = train.sum(axis=1)
test["sum"] = test.sum(axis=1)
train["var"] = tmp_train.var(axis=1)
test["var"] = tmp_test.var(axis=1)
train["median"] = tmp_train.median(axis=1)
test["median"] = tmp_test.median(axis=1)
train["mean"] = tmp_train.mean(axis=1)
test["mean"] = tmp_test.mean(axis=1)
train["std"] = tmp_train.std(axis=1)
test["std"] = tmp_test.std(axis=1)
train["max"] = tmp_train.max(axis=1)
test["max"] = tmp_test.max(axis=1)
train["min"] = tmp_train.min(axis=1)
test["min"] = tmp_test.min(axis=1)
del(tmp_train)
del(tmp_test)

# train data is valid , test data has nan and infinite
tmp = pd.DataFrame(np.nan_to_num(tmp))
# Go through the columns one at a time (can't do it all at once for this dataset)
total_df = deepcopy(tmp)      
print('np.any(np.isnan(total_df)', np.any(np.isnan(total_df)))
print('np.all(np.isfinite(total_df)', np.all(np.isfinite(total_df)))

p = progressbar.ProgressBar()
p.start()

# Mean-variance scale all columns excluding 0-values'
print('total_df.columns:',total_df.columns) 
columnsCount = len(total_df.columns)
for col in total_df.columns:    
    p.update(col/columnsCount * 100)

    # Detect outliers in this column
    data = total_df[col].values
    data_mean, data_std = np.mean(data), np.std(data)
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    outliers = [x for x in data if x < lower or x > upper]
    
    # If there are crazy high values, do a log-transform
    if len(outliers) > 0:
        non_zero_idx = data != 0
        total_df.loc[non_zero_idx, col] = np.log(data[non_zero_idx])
    
    # Scale non-zero column values
    nonzero_rows = total_df[col] != 0
    if  np.isfinite(total_df.loc[nonzero_rows, col]).all():
        total_df.loc[nonzero_rows, col] = scale(total_df.loc[nonzero_rows, col])
        if  np.isfinite(total_df[col]).all():
            # Scale all column values
            total_df[col] = scale(total_df[col])
    gc.collect()
    
p.finish()

NUM_OF_COM = 100 #need tuned
transformer = random_projection.SparseRandomProjection(n_components = NUM_OF_COM)
RP = transformer.fit_transform(tmp)
rp = pd.DataFrame(RP)
columns = ["RandomProjection{}".format(i) for i in range(NUM_OF_COM)]
rp.columns = columns

rp_train = rp[:ntrain]
rp_test = rp[ntrain:]
rp_test.index = test.index

#concat RandomProjection and raw data
train = pd.concat([train,rp_train],axis=1)
test = pd.concat([test,rp_test],axis=1)

del(rp_train)
del(rp_test)
train.shape

#define evaluation method for a given model. we use k-fold cross validation on the training set. 
#the loss function is root mean square logarithm error between target and prediction
#note: train and y_train are feeded as global variables
NUM_FOLDS = 5 #need tuned
def rmsle_cv(model):
    kf = KFold(NUM_FOLDS, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.055, colsample_bylevel =0.5, 
                             gamma=1.5, learning_rate=0.02, max_depth=32, 
                             objective='reg:linear',booster='gbtree',
                             min_child_weight=57, n_estimators=1000, reg_alpha=0, 
                             reg_lambda = 0,eval_metric = 'rmse', subsample=0.7, 
                             silent=1, n_jobs = -1, early_stopping_rounds = 14,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=144,
                              learning_rate=0.005, n_estimators=720, max_depth=13,
                              metric='rmse',is_training_metric=True,
                              max_bin = 55, bagging_fraction = 0.8,verbose=-1,
                              bagging_freq = 5, feature_fraction = 0.9)  

#ensemble method: model averaging
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    # the reason of clone is avoiding affect the original base models
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]  
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([ model.predict(X) for model in self.models_ ])
        return np.mean(predictions, axis=1)

#cross validation   
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
averaged_models = AveragingModels(models = (model_xgb, model_lgb))
score = rmsle_cv(averaged_models)
print("averaged score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

averaged_models.fit(train.values, y_train)
pred = np.expm1(averaged_models.predict(test.values))

############################################################################## CatBoost + decomposition features
# Specify index/ target name
id_col = "ID"
target_var = "target"

# House Keeping Parameters
Debug = False
Home = False
Build_Results_csv = False # if running for first time

# Read train and test files
train_full = pd.read_csv('train.csv', index_col = id_col)
traindex = train_full.index

test_full = pd.read_csv('test.csv', index_col = id_col)
traindex = test_full.index

# Remove columns with a std of 0
zero_std_cols = train_full.columns[train_full.std() == 0]
train_full.drop(zero_std_cols, axis=1, inplace=True)
test_full.drop(zero_std_cols, axis=1, inplace=True)
print("Removed %s constant columns") % len(zero_std_cols)

# Remove duplicate columns
colsToRemove = []
colsScaned = []
dupList = {}
columns = train_full.columns
for i in range(len(columns)-1):
    v = train_full[columns[i]].values
    dupCols = []
    for j in range(i+1,len(columns)):
        if np.array_equal(v, train_full[columns[j]].values):
            colsToRemove.append(columns[j])
            if columns[j] not in colsScaned:
                dupCols.append(columns[j]) 
                colsScaned.append(columns[j])
                dupList[columns[i]] = dupCols
colsToRemove = list(set(colsToRemove))
train_full.drop(colsToRemove, axis=1, inplace=True)
test_full.drop(colsToRemove, axis=1, inplace=True)
print("Dropped %s duplicate columns") % len(colsToRemove)

changed_type = []
for col, dtype in train_full.dtypes.iteritems():
    if dtype==np.int64:
        max_val = np.max(train_full[col])
        bits = np.log(max_val)/np.log(2)
        if bits < 8:
            new_dtype = np.uint8
        elif bits < 16:
            new_dtype = np.uint16
        elif bits < 32:
            new_dtype = np.uint32
        else:
            new_dtype = None
        if new_dtype:
            changed_type.append(col)
            train_full[col] = train_full[col].astype(new_dtype)
print('Changed types on {} columns'.format(len(changed_type)))

sparsity = {
    col: (train_full[col] == 0).mean()
    for idx, col in enumerate(train_full)
}
sparsity = pd.Series(sparsity)

# 2. CatBoost + decomposition features

print("Load data...")
train = train_full.copy(deep=True)
test = test_full.copy(deep=True)
target = np.log1p(train['target']).values
subm = pd.read_csv('sample_submission.csv')
print("Train shape: {}\nTest shape: {}".format(train.shape, test.shape))
col = [c for c in train.columns if c not in ['target']]

scl = preprocessing.StandardScaler()
def rmsle(y, pred):
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(pred), 2)))

x1, x2, y1, y2 = model_selection.train_test_split(train[col], train.target.values, test_size=0.10, random_state=5)
model = RandomForestRegressor(n_jobs = -1, random_state = 7)
model.fit(scl.fit_transform(x1), y1)
print(rmsle(y2, model.predict(scl.transform(x2))))

col = pd.DataFrame({'importance': model.feature_importances_, 'feature': col}).sort_values(by=['importance'], ascending=[False])[:600]['feature'].values

#Added Columns from feature_selection
train = train[['target']+list(col)]
test = test[list(col)]
print("Train shape: {}\nTest shape: {}".format(train.shape, test.shape))

N_COMP = 20            ### Number of decomposition components ###

print("Define training features...")
exclude_other = ['ID', 'target']
train_features = []
for c in train.columns:
    #if c not in cols_to_drop \
    #and c not in exclude_other:
    if c not in exclude_other:
        train_features.append(c)
print("Number of featuress for training: %s" % len(train_features))

train, test = train[train_features], test[train_features]
print("\nTrain shape: {}\nTest shape: {}".format(train.shape, test.shape))

print("\nStart decomposition process...")
print("PCA")
pca = PCA(n_components=N_COMP, random_state=17)
pca_results_train = pca.fit_transform(train)
pca_results_test = pca.transform(test)

print("tSVD")
tsvd = TruncatedSVD(n_components=N_COMP, random_state=17)
tsvd_results_train = tsvd.fit_transform(train)
tsvd_results_test = tsvd.transform(test)

print("ICA")
ica = FastICA(n_components=N_COMP, random_state=17)
ica_results_train = ica.fit_transform(train)
ica_results_test = ica.transform(test)

print("GRP")
grp = GaussianRandomProjection(n_components=N_COMP, eps=0.1, random_state=17)
grp_results_train = grp.fit_transform(train)
grp_results_test = grp.transform(test)

print("SRP")
srp = SparseRandomProjection(n_components=N_COMP, dense_output=True, random_state=17)
srp_results_train = srp.fit_transform(train)
srp_results_test = srp.transform(test)

print("Append decomposition components to datasets...")
for i in range(1, N_COMP + 1):
    train['pca_' + str(i)] = pca_results_train[:, i - 1]
    test['pca_' + str(i)] = pca_results_test[:, i - 1]

    train['ica_' + str(i)] = ica_results_train[:, i - 1]
    test['ica_' + str(i)] = ica_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]
print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))

print('\nModelling...')
def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))

folds = KFold(n_splits=5, shuffle=True, random_state=546789)
oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test.shape[0])

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train)):
    trn_x, trn_y = train.ix[trn_idx], target[trn_idx]
    val_x, val_y = train.ix[val_idx], target[val_idx]
    cb_model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=4, l2_leaf_reg=20, bootstrap_type='Bernoulli', subsample=0.6, eval_metric='RMSE', metric_period=50, od_type='Iter', od_wait=45, random_seed=17, allow_writing_files=False)
    cb_model.fit(trn_x, trn_y, eval_set=(val_x, val_y), cat_features=[], use_best_model=True, verbose=True)
    oof_preds[val_idx] = cb_model.predict(val_x)
    sub_preds += cb_model.predict(test) / folds.n_splits
    print("Fold %2d RMSLE : %.6f" % (n_fold+1, rmsle(np.exp(val_y)-1, np.exp(oof_preds[val_idx])-1)))

print("Full RMSLE score %.6f" % rmsle(np.exp(target)-1, np.exp(oof_preds)-1)) 
cb_ans = np.exp(sub_preds) - 1

############################################################## Final visualizations and solution processing

# Find correlation between ensemble models
print('merging..')
dataframe=pd.DataFrame({'CB': cb_ans, 'LGB_XGB': pred})
corr = dataframe.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

# View a distribution of solutions
plot_frame=dataframe.reset_index()
plot_frame = plot_frame[:100]
ax = plot_frame.plot.scatter(x='index', y='CB', color='r', label='CB')
plot_frame.plot.scatter(x='index', y='LGB_XGB', color='b', ax=ax, label ='LGB_XGB')
ax.set_xlabel('index')
ax.set_ylabel('target')
plt.show

# Combine model results and write submission file
ensemble_ans = (cb_ans + pred) / 2

subm = pd.read_csv('sample_submission.csv')

subm['target'] = ensemble_ans

subm.to_csv('submission.csv',index=False)