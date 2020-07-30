# Errors

## Input contains NaN, infinity or a value too large for dtype('float64').

下記で発生．

```
lasso.predict(X_test)
```

### 確認したこと

```
X_test = load_x(features, dropped_ids)['test']
print_exit(X_test.isnull().all())  # => all false
print_exit(np.isinf(X_test).all())  # => all false
print_exit(X_test.describe())  # => count of GarageArea, TotalBsmtSF are 1458, others are 1459.
df_X['col'] = np.log1p(df_X['col])  # => raise error
```

下記で解決．

```
X_test = X_test.fillna(X_test.mean())
```
