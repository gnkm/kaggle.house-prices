import copy
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings


# warnings.simplefilter('ignore')

# Don't define any function in this file,
# thus don't define main function.
features = [
    'OverallQual',
    'GrLivArea',
    'GarageArea',
    'TotalBsmtSF',
    # Added for getting normality
    'HasGarage',
    'HasBsmt',
]
col_id_name = 'Id'
col_target_name = 'SalePrice'
df_train = pd.read_feather('data/input/train.feather')
df_train['HasGarage'] = pd.Series(
    len(df_train['GarageArea']),
    index=df_train.index
)
df_train['HasGarage'] = 0
df_train.loc[df_train['GarageArea'] > 0, 'HasGarage'] = 1
df_train['HasBsmt'] = pd.Series(
    len(df_train['TotalBsmtSF']),
    index=df_train.index
)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
all_features = copy.deepcopy(features)
all_features.extend([col_id_name, col_target_name])
df_train = df_train[all_features]
# Dealing with missing data(Drop rows)
df_train = df_train.dropna(axis='index')
# Dealing with outlier: Refer to EDA
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
# Transform for getting normality
df_train['SalePrice'] = np.log(df_train['SalePrice'])
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
df_train.loc[df_train['HasGarage'] == 1, 'GarageArea'] \
    = np.log(df_train['GarageArea'])
df_train.loc[df_train['HasBsmt'] == 1, 'TotalBsmtSF'] \
    = np.log(df_train['TotalBsmtSF'])
X_train = df_train[features]
y_train = df_train[col_target_name]
estimator = RidgeCV(
    alphas=[i / 10 for i in range(1, 10)],
    cv=10,
    scoring='r2',
)
estimator.fit(X_train, y_train)
print(estimator.score(X_train, y_train)) # score: 0.8278921565126609
