import argparse
from datetime import datetime as dt
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import yaml

# from models import lgbm as my_lgbm
from cv import r2_cv
from param_tuning.optimizer import LGBMRegressorOptimizer
from preprocessing import load_x, load_y
from utils import print_exit, print_float


# Don't define any function in this file,
# thus don't define main function.

# use var `now` in config file and submit file.
now = dt.now().strftime('%Y-%m-%d-%H-%M-%S')

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.yml')
options = parser.parse_args()

with open(options.config, 'r') as file:
    config = yaml.safe_load(file)

features = config['extracted_features']
col_id_name = config['col_id_name']
col_target_name = config['col_target_name']
dropped_ids = config['dropped_ids']
lgbm_params = config['lgbm_params']
n_folds = config['cv']['n_folds']
param_candidates = config['lgbm_params']['candidates']

Xs = load_x(features, dropped_ids)
X_train_all = Xs['train']
X_test = Xs['test']
y_train_all = load_y(col_id_name, col_target_name, dropped_ids)

# @todo: Modify preprocessor
X_test = X_test.fillna(X_test.mean())

# Stacking
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

estimators = [
    ('lasso', lasso),
    ('ENet', ENet),
    ('KRR', KRR),
    ('GBoost', GBoost),
]
regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=RandomForestRegressor(n_estimators=10,
                                          random_state=42)
)
# Train
regressor.fit(X_train_all, y_train_all)
# Predict
y_pred_logarithmic = regressor.predict(X_test)  # error
y_pred = np.exp(y_pred_logarithmic)
# Evaluate
scores = r2_cv(regressor, X_train_all, y_train_all, n_folds)

score = scores.mean()

sub_df = pd.DataFrame(
    pd.read_feather('data/input/test.feather')[col_id_name]
)
sub_df[col_target_name] = y_pred
sub_df.to_csv(
    './data/output/sub_{time}_{score:.5f}.csv'.format(
        time=now,
        score=score,
    ),
    index=False
)

config_file_name = './configs/{time}_{score:.5f}.yml'.format(
    time=now,
    score=score,
)
with open(config_file_name, 'w') as file:
    yaml.dump(config, file)
