import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split


def r2_cv(model, X_train, y_train, n_folds):
    kf = KFold(
        n_folds,
        shuffle=True,
        random_state=42
    )
    scores = np.sqrt(
        cross_val_score(
            model,
            X_train,
            y_train,
            scoring='r2',
            cv = kf,
        )
    )
    return scores
