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
from param_tuning.optimizer import ENetOptimizer, LassoOptimizer, LGBMRegressorOptimizer
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
random_state = config['random_state']
n_folds = config['cv']['n_folds']
hyper_parameters = config['params']

Xs = load_x(features, dropped_ids)
X_train_all = Xs['train']
X_test = Xs['test']
y_train_all = load_y(col_id_name, col_target_name, dropped_ids)

# @todo: Modify preprocessor
X_test = X_test.fillna(X_test.mean())

# Lasso
lasso_with_param_candidates = make_pipeline(
    RobustScaler(),
    Lasso(
        random_state=random_state
    )
)
lasso_optimizer = LassoOptimizer(
    lasso_with_param_candidates,
    X_train_all,
    y_train_all,
    n_folds,
    hyper_parameters['lasso']['candidates'],
)
lasso_best_params = lasso_optimizer.optimize()
lasso = make_pipeline(
    RobustScaler(),
    Lasso(
        alpha=lasso_best_params['lasso__alpha'],
        random_state=random_state,
    )
)

# Elasticnet
enet_with_param_candidates = make_pipeline(
    RobustScaler(),
    ElasticNet(
        random_state=random_state
    )
)
enet_optimizer = ENetOptimizer(
    enet_with_param_candidates,
    X_train_all,
    y_train_all,
    n_folds,
    hyper_parameters['enet']['candidates']
)
enet_best_params = enet_optimizer.optimize()
ENet = make_pipeline(
    RobustScaler(),
    ElasticNet(
        alpha=enet_best_params['elasticnet__alpha'],
        l1_ratio=enet_best_params['elasticnet__l1_ratio'],
        random_state=random_state,
    )
)
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=4,
    max_features='sqrt',
    min_samples_leaf=15,
    min_samples_split=10,
    loss='huber',
    random_state =5
)

lgbm_instance_params = hyper_parameters['lgbm']['instance']
lgbm_regressor_with_param_candidates = LGBMRegressor(
    random_state=random_state,
    silent=lgbm_instance_params['silent'],
)

lgbm_optimizer = LGBMRegressorOptimizer(
    lgbm_regressor_with_param_candidates,
    X_train_all,
    y_train_all,
    n_folds,
    hyper_parameters['lgbm']['candidates']
)
lgbm_best_params = lgbm_optimizer.optimize()

lgbm_regressor_with_optimized_params = LGBMRegressor(
    boosting_type=lgbm_best_params['boosting_type'],
    learning_rate=lgbm_best_params['learning_rate'],
    lambda_l1=lgbm_best_params['lambda_l1'],
    lambda_l2=lgbm_best_params['lambda_l2'],
    # default params
    random_state=random_state,
    silent=lgbm_instance_params['silent'],
)
lgbm_regressor_with_optimized_params.fit(X_train_all, y_train_all)
lgbm_y_pred_logarithmic = lgbm_regressor_with_optimized_params.predict(X_test)  # error
lgbm_y_pred = np.exp(lgbm_y_pred_logarithmic)

# Stacking
estimators = [
    ('lasso', lasso),
    ('ENet', ENet),
    ('KRR', KRR),
    ('GBoost', GBoost),
    ('LGBM', lgbm_regressor_with_optimized_params),
]
stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=RandomForestRegressor(
        n_estimators=10,
        random_state=42
    )
)
# Train
stacking_regressor.fit(X_train_all, y_train_all)
# Predict
y_pred_logarithmic = stacking_regressor.predict(X_test)  # error
y_pred = np.exp(y_pred_logarithmic)
# Evaluate
scores = r2_cv(stacking_regressor, X_train_all, y_train_all, n_folds)

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
