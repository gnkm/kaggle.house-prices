"""Class for optimizing hyper parameters.

This class wraps optuna api.

Example:
    from param_tuning.optimizer import Optimizer

    model = something_estimator()
    prams = {'param_1': v_1, ... , 'param_n', v_n}
    optimizer = Optimizer(model, X_train, y_train, n_folds, param_candidates)
    best_params = optimizer.optimize()

Todo:
    * Implement interfaces like sklearn GridSearchCV
"""

from abc import ABCMeta, abstractmethod
import optuna

from cv import r2_cv


class BaseOptimizer(metaclass=ABCMeta):
    """Base optimizer class.

    Optimizer of estimator class is a inheritence class of this.

    Args:
        n_folds (int): fold number of `kFold`, not `cross_val_score`.
    """

    # def __init__(self, model, X_train, y_train, n_folds, param_candidates):
    def __init__(self, model, X_train, y_train, n_folds, param_candidates):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.n_folds = n_folds
        self.param_candidates = param_candidates

    def optimize(self):
        """Return optimized parameters.
        """
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=100)
        return study.best_params

    @abstractmethod
    def objective(self, trial):
        raise NotImplementedError


class ENetOptimizer(BaseOptimizer):
    """Optimizer for ElasticNet.

    Function `objective` is implemented in this class.
    """
    def objective(self, trial):
        alpha = trial.suggest_loguniform(
            'elasticnet__alpha',
            self.param_candidates['alpha']['low'],
            self.param_candidates['alpha']['high'],
        )
        l1_ratio = trial.suggest_float(
            'elasticnet__l1_ratio',
            self.param_candidates['l1_ratio']['low'],
            self.param_candidates['l1_ratio']['high'],
            step=self.param_candidates['l1_ratio']['step'],
        )
        model = self.model
        model.set_params(
            elasticnet__alpha=alpha,
            elasticnet__l1_ratio=l1_ratio,
        )
        X_train = self.X_train
        y_train = self.y_train
        n_folds = self.n_folds
        return r2_cv(model, X_train, y_train, n_folds).mean()


class LassoOptimizer(BaseOptimizer):
    """Optimizer for Lasso.

    Function `objective` is implemented in this class.
    """
    def objective(self, trial):
        alpha = trial.suggest_loguniform(
            'lasso__alpha',
            self.param_candidates['alpha']['low'],
            self.param_candidates['alpha']['high'],
        )
        model = self.model
        model.set_params(
            lasso__alpha=alpha
        )
        X_train = self.X_train
        y_train = self.y_train
        n_folds = self.n_folds
        return r2_cv(model, X_train, y_train, n_folds).mean()


class LGBMRegressorOptimizer(BaseOptimizer):
    """Optimizer for LGBMRegressor.

    Function `objective` is implemented in this class.
    """
    def objective(self, trial):
        boosting_type = trial.suggest_categorical(
            'boosting_type',
            self.param_candidates['boosting_type']
        )
        learning_rate = trial.suggest_loguniform(
            'learning_rate',
            self.param_candidates['learning_rate']['low'],
            self.param_candidates['learning_rate']['high'],
        )
        lambda_l1 = trial.suggest_float(
            'lambda_l1',
            self.param_candidates['lambda_l1']['low'],
            self.param_candidates['lambda_l1']['high'],
            step=self.param_candidates['lambda_l1']['step'],
        )
        lambda_l2 = trial.suggest_float(
            'lambda_l2',
            self.param_candidates['lambda_l2']['low'],
            self.param_candidates['lambda_l2']['high'],
            step=self.param_candidates['lambda_l2']['step'],
        )
        model = self.model
        model.set_params(
            boosting_type=boosting_type,
            learning_rate=learning_rate,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
        )
        X_train = self.X_train
        y_train = self.y_train
        n_folds = self.n_folds
        return r2_cv(model, X_train, y_train, n_folds).mean()
