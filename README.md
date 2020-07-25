# kaggle.house-prices

## Kaggle Page

[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## Reference

- [COMPREHENSIVE DATA EXPLORATION WITH PYTHON | Kaggle](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)
- [Stacked Regressions : Top 4% on LeaderBoard | Kaggle](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)

## Usage

### Pull docker image

```
docker pull gcr.io/kaggle-images/python
```

### Run Jupyter

```
./docker-run.sh jp
```

### Run Python

#### Convert CSV to Feather

```
./docker-run.sh py scripts/convet_to_feather.py
```

#### Main Script

```
./docker-run.sh py run.py
```

## For Development with VSCode

- Install `Remote - Container` package
- Start jupyter container
- Input "Remote-Containers: Attach to running container" to command palette
- Select jupyter container
- Select python interpreter: Python 3.7.6 64-bit ('base': conda)
