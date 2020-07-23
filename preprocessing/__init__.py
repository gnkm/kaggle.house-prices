import copy
import numpy as np
import pandas as pd

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
dropped_ids = [524, 1299]

def load_x():
    """Return dict contains df_X_train and df_X_test.
    
    Args:
        features: List
        col_id_name: str
        col_target_name: str
    
    Returns:
        Dict
            {
                'train': df_X_train,
                'test': df_X_test
            }
    """
    dfs = {}
    dfs['train'] = pd.read_feather('data/input/train.feather')
    dfs['test'] = pd.read_feather('data/input/test.feather')
    for key, df in dfs.items():
        # Add some new features
        df['HasGarage'] = pd.Series(
            len(df['GarageArea']),
            index=df.index
        )
        df['HasGarage'] = 0
        df.loc[df['GarageArea'] > 0, 'HasGarage'] = 1
        df['HasBsmt'] = pd.Series(
            len(df['TotalBsmtSF']),
            index=df.index
        )
        df['HasBsmt'] = 0
        df.loc[df['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
        # Transform for getting normality
        df['GrLivArea'] = np.log(df['GrLivArea'])
        df.loc[df['HasGarage'] == 1, 'GarageArea'] \
            = np.log(df['GarageArea'])
        df.loc[df['HasBsmt'] == 1, 'TotalBsmtSF'] \
            = np.log(df['TotalBsmtSF'])
        # Use `deepcopy()` for making var `feature` immutable.
        # If `feature` is mutable, 'SalePrice' is added when key is 'train'.
        # And 'SalePrice' exists in `feature`, when key is 'test'.
        # This occures error: 'SalePrice' is not index.
        needed_features = copy.deepcopy(features)
        if key == 'train':
            needed_features.extend([col_id_name, col_target_name])
            df = _preprocess_train(df[needed_features])
        elif key == 'test':
            needed_features.extend([col_id_name])
        df = df[needed_features]
        dfs[key] = df

    return dfs

def load_y():
    df = pd.read_feather('data/input/train.feather')
    df = df[[col_id_name, col_target_name]]
    df = _drop_outlier_by_id(df)
    return df[col_target_name]

def _preprocess_train(df):
    # Transform for getting normality
    # Use df.copy().
    # If don't use, following warning occur.
    # > A value is trying to be set on a copy of a slice from a DataFrame.
    # > Try using .loc[row_indexer,col_indexer] = value instead
    # df['SalePrice'] = np.log(df['SalePrice'])  # => warning
    df_copied = df.copy()
    df_copied['SalePrice'] = np.log(df_copied['SalePrice'])
    # # Dealing with missing data(Drop rows)
    df_copied = df_copied.dropna(axis='index')
    # Dealing with outlier: Refer to EDA
    df_copied = _drop_outlier_by_id(df_copied)
    return df_copied

    return df_copied

def _drop_outlier_by_id(df):
    for i in dropped_ids:
        df.drop(df[df['Id'] == i].index)
    return df