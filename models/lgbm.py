import lightgbm as lgb


def train(X_train, X_valid, y_train, y_valid, lgbm_params):
    """Return trained model.

    Args:
        X_train: DataFrame
        X_valid: DataFrame
        y_train: DataFrame
        y_valid: DataFrame
        lgbm_params: Dict

    Returns:
        lightgbm.Booster
    """
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    model = lgb.train(
        lgbm_params,
        lgb_train,
        valid_sets=lgb_eval,
        num_boost_round=1000,
        early_stopping_rounds=10,
    )
    return model

def cv():
    """Implement this function when needed.

    Referance: https://www.kaggle.com/kenmatsu4/using-trained-booster-from-lightgbm-cv-w-callback
    """
    pass

def predict(model, X_test):
    """Return predict list.

    Args:
        model: lightgbm.Booster
        X_test: DataFrame

    Returns:
        np.ndarray
    """
    y_pred = model.predict(X_test)
    return y_pred
