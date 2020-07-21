import copy
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings


# warnings.simplefilter('ignore')


def main():
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

    # Cross validation and grid search
    param_grid = [
        {'Ridge__alpha': [0.5, 0.6, 0.7, 0.8, 0.9, 1]},
    ]
    gs = GridSearchCV(
        estimator=Ridge(random_state=1),
        param_grid=param_grid,
        scoring='r2',
        cv=2,
    )
    scores = cross_val_score(
        gs,
        X_train,
        y_train,
        scoring='r2',
        cv=5,
    )
    print(scores)


if __name__ == '__main__':
    main()
