
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

Xs = load_x(features, dropped_ids)
X_train_all = Xs['train']
X_test = Xs['test']
y_train_all = load_y(col_id_name, col_target_name, dropped_ids)

reg_params = lgbm_params['instance']
regressor = LGBMRegressor(
    boosting_type=reg_params['boosting_type'],
    learning_rate=reg_params['learning_rate'],
    reg_alpha=reg_params['reg_alpha'],
    reg_lambda=reg_params['reg_lambda'],
    random_state=reg_params['random_state'],
    silent=reg_params['silent'],
)

scores = r2_cv(regressor, X_train_all, y_train_all, n_folds)
score = scores.mean()
print_exit(
    print_float(score)
)

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

config_file_name = './configs/{time}_{score:.5f}.yml'.format(
    time=now,
    score=score,
)
with open(config_file_name, 'w') as file:
    yaml.dump(config, file)
