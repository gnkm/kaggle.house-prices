import numpy as np
import pandas as pd


def load_x(features, dropped_ids):
    """Return dict contains df_X_train and df_X_test.
    
    Args:
        features: List
        dropped_ids: List
    
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
        if key == 'train':
            df = _drop_outlier_by_id(df, dropped_ids)
        df = df[features]
        dfs[key] = df

    return dfs

def load_y(col_id_name, col_target_name, dropped_ids):
    df = pd.read_feather('data/input/train.feather')
    df = df[[col_id_name, col_target_name]]
    df = _drop_outlier_by_id(df, dropped_ids)
    # Transform for getting normality
    df[col_target_name] = np.log(df[col_target_name])
    return df[col_target_name]

def _drop_outlier_by_id(df, dropped_ids):
    for i in dropped_ids:
        df.drop(df[df['Id'] == i].index)
    return df