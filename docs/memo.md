# Memo

## Todo

- アンサンブル学習で予測する

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
