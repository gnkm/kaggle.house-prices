
import argparse
from datetime import datetime as dt
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
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

reg_params = lgbm_params['instance']
regressor = LGBMRegressor(
    boosting_type=reg_params['boosting_type'],
    learning_rate=reg_params['learning_rate'],
    # reg_alpha=reg_params['reg_alpha'],
    reg_lambda=reg_params['reg_lambda'],
    random_state=reg_params['random_state'],
    silent=reg_params['silent'],
)

# cv_scores = r2_cv(regressor, X_train_all, y_train_all, n_folds)
# cv_score = cv_scores.mean()

# Optimize training data ratio
# i.e. Search best n_folds
# @todo: Modulize
from sklearn.model_selection import learning_curve

train_sizes = range(0.1, 1, 0.1)
kf = KFold(n_splits=5)
train_sizes, train_scores, valid_scores \
    = learning_curve(X_train_all, y_train_all, train_sizes, cv=kf)
# @todo: Implement f
best_n_folds = f(train_sizes, train_scores, valid_scores)
# optimizer = LGBMRegressorOptimizer(regressor, X_train_all, y_train_all, best_n_folds, param_candidates)

optimizer = LGBMRegressorOptimizer(regressor, X_train_all, y_train_all, n_folds, param_candidates)
best_params = optimizer.optimize()
print_exit(best_params)

# Train
regressor.fit(X_train_all, y_train_all)
# Predict
y_pred_logarithmic = regressor.predict(X_test)
y_pred = np.exp(y_pred_logarithmic)

# Evaluate
y_pred_from_train = regressor.predict(X_train_all)
score = r2_score(y_train_all, y_pred_from_train)

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
