
from datetime import datetime as dt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from models import lgbm as my_lgbm
from preprocessing import load_x, load_y
from utils import print_exit


# Don't define any function in this file,
# thus don't define main function.

# use var `now` in config file and submit file.
now = dt.now().strftime('%Y-%m-%d-%H-%M-%S')

# @todo: manage config with external file
config = {
    'extracted_features': [
        'OverallQual',
        'GrLivArea',
        'GarageArea',
        'TotalBsmtSF',
        # Added for getting normality
        'HasGarage',
        'HasBsmt',
    ],
    'col_id_name': 'Id',
    'col_target_name': 'SalePrice',
    'dropped_ids': [524, 1299],
}

features = config['extracted_features']
col_id_name = config['col_id_name']
col_target_name = config['col_target_name']
dropped_ids = config['dropped_ids']

Xs = load_x(features, dropped_ids)
X_train_all = Xs['train']
X_test = Xs['test']
y_train_all = load_y(col_id_name, col_target_name, dropped_ids)

r2s_valid = []
y_preds = []
models = []

# @todo: Define params
# Reference: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
lgbm_params = {
    'max_bin': 100,
    'learning_rate': 0.1,
    'num_itarations': 100,
    'num_leaves': 31,
    'boosting': 'dart',
    'objective': 'regression',
    'metric': 'rmse',
    'lambda_l1': 0,
    'lambda_l2': 0,
    'verbosity': -1,
}

# Debug(for faster speed)
# lgbm_params = {
#     'max_bin': 10,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 3,
#     'save_binary': True,
#     'objective': 'regression',
#     'metric': 'rmse',
#     'lambda_l1': 0,
#     'lambda_l2': 0,
#     'verbosity': -1,
# }

kf = KFold(n_splits=10)
for train_index, valid_index in kf.split(X_train_all):
    X_train, X_valid = X_train_all.iloc[train_index, :], X_train_all.iloc[valid_index, :]
    y_train, y_valid = y_train_all.iloc[train_index], y_train_all.iloc[valid_index]

    # Train
    model_trained = my_lgbm.train(X_train, X_valid, y_train, y_valid, lgbm_params)
    # Calculate r2 score
    y_pred_from_train = my_lgbm.predict(model_trained, X_train)
    y_pred_from_valid = my_lgbm.predict(model_trained, X_valid)
    r2_valid = r2_score(y_valid, y_pred_from_valid)
    r2s_valid.append(r2_valid)
    # Predict
    y_pred_logarithmic = my_lgbm.predict(model_trained, X_test)
    # Target var is transformed with `np.log()`
    y_pred = np.exp(y_pred_logarithmic)
    y_preds.append(y_pred)
    models.append(model_trained)

score = np.mean(r2s_valid)
pred_avg = np.mean(y_preds, axis=0)

sub_df = pd.DataFrame(
    pd.read_feather('data/input/test.feather')[col_id_name]
)
sub_df[col_target_name] = pred_avg
sub_df.to_csv(
    './data/output/sub_{time}_{score:.5f}.csv'.format(
        time=now,
        score=score,
    ),
    index=False
)
