# Memo

## Todo

- bagging
  - その場合の評価
- Predict with deep learning
- Logging: 時間のかかっている処理を明らかにする
- Separate param tuning script
- `optimizer.optimize()` で得たパラメータを明示する(`config.yml` に保存するなど)
- pdpipe で前処理のフローをわかりやすくする

## Consider

### Optimize Parameters of Stacked Estimators

regressor ごとにチューニングしてから stacking する．

### Invalid Parameters Error

> ValueError: Invalid parameter ENet for estimator Pipeline
> Check the list of available parameters with `estimator.get_params().keys()`.

`optimizer.py` の中のパラメータ名称について．
下記例の `'elasticnet__l1_ratio'` は
`model` を定義した後，`model.get_params().keys()` で確認する．

```
        l1_ratio = trial.suggest_float(
            'elasticnet__l1_ratio',
            self.param_candidates['l1_ratio']['low'],
            self.param_candidates['l1_ratio']['high'],
            step=self.param_candidates['l1_ratio']['step'],
        )
        model = self.model
        print(model.get_params().keys())
```

### Preprocessing

Do preprocess of X_train, X_test, and y_train at once.

```
df_train = pd.read_feather('')
df_test = pd.read_feather('')
preprocessed_data = load_data(df_train, df_test, col_id_name, col_target_name, dropped_ids)
X_train_all = preprocessed_data['X_train']
X_test = preprocessed_data['X_test']
y_train_all = preprocessed_data['y_train']
```

### Optimize training data ratio with learning_curve()

=> try 3, 5, 10 for `n_folds`.

```
# Optimize training data ratio
# i.e. Search best n_folds
# @todo: Modulize
from sklearn.model_selection import learning_curve

train_sizes = range(0.1, 1, 0.1)
kf = KFold(n_splits=5)
train_sizes, train_scores, valid_scores \
    = learning_curve(X_train_all, y_train_all, train_sizes, cv=kf)
# @todo: Implement f
best_n_folds = f(train_sizes, train_scores, valid_scores)
optimizer = LGBMRegressorOptimizer(regressor, X_train_all, y_train_all, best_n_folds, param_candidates)
```

## Study

### Stacked Regressions : Top 4% on LeaderBoard

[Stacked Regressions : Top 4% on LeaderBoard | Kaggle](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)

前処理は最初に

```
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
```

としてから行う．最後に

```
train = all_data[:ntrain]
test = all_data[ntrain:]
```

として，train と test にわける．

- 全ての特徴量を使い，欠損値処理を行う
- skew の高い特徴量に対し，Box-Cox 変換で normalize する

## LGBM

- [Python API — LightGBM 2.3.2 documentation](https://lightgbm.readthedocs.io/en/latest/Python-API.html)
- [Parameters — LightGBM 2.3.2 documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html)

## Optuna

- [API Reference — Optuna 2.0.0 documentation](https://optuna.readthedocs.io/en/stable/reference/index.html)

### Trial.suggest_*()

[Trial — Optuna 1.5.0 documentation](https://optuna.readthedocs.io/en/stable/reference/trial.html)

- suggest_categorical(name, choices)
- suggest_discrete_uniform(name, low, high, q: int)
- suggest_float(name: str, low: float, high: float, *, step: Optional[float] = None, log: bool = False) -> float
- suggest_int(name, low, high, step=1, log=False)
- suggest_loguniform(name, low, high)
- suggest_uniform(name, low, high)
