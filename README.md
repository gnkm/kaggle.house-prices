# kaggle.house-prices

## Kaggle Page

[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

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
