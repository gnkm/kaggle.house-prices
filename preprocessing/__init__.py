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
        if key == 'train':
            df = _drop_outlier_by_id(df)
        df = df[features]
        dfs[key] = df

    return dfs

def load_y():
    df = pd.read_feather('data/input/train.feather')
    df = df[[col_id_name, col_target_name]]
    df = _drop_outlier_by_id(df)
    return df[col_target_name]

def _drop_outlier_by_id(df):
    for i in dropped_ids:
        df.drop(df[df['Id'] == i].index)
    return df