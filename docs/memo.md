# Memo

## Todo

- バイアスとバリアンスの評価
- アンサンブル学習で予測する

## Consider

### Preprocessing

`run.py`

Whether preprocess X_train, X_test, and y_train at once.

```
df_train = pd.read_feather('')
df_test = pd.read_feather('')
preprocessed_data = load_data(df_train, df_test, col_id_name, col_target_name, dropped_ids)
X_train_all = preprocessed_data['X_train']
X_test = preprocessed_data['X_test']
y_train_all = preprocessed_data['y_train']
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

## Optuna

### Trial.suggest_*()

- suggest_categorical(name, choices)
- suggest_discrete_uniform(name, low, high, q: int)
- suggest_float(name: str, low: float, high: float, *, step: Optional[float] = None, log: bool = False) -> float
- suggest_int(name, low, high, step=1, log=False)
- suggest_loguniform(name, low, high)
- suggest_uniform(name, low, high)

reference: [Trial — Optuna 1.5.0 documentation](https://optuna.readthedocs.io/en/stable/reference/trial.html)
