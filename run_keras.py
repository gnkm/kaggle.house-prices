import argparse
from datetime import datetime as dt
import numpy as np
import yaml
from keras.layers import Dense
from keras.models import Sequential
from keras.losses import MeanSquaredError
from sklearn.metrics import r2_score


from preprocessing import load_x, load_y
from utils import print_exit


# Don't define any function in this file,
# thus don't define main function.

# use var `now` in config file and submit file.
now = dt.now().strftime('%Y-%m-%d-%H-%M-%S')

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.yml')
options = parser.parse_args()

with open(options.config, 'r') as file:
    config = yaml.safe_load(file)

features = config['extracted_features']
col_id_name = config['col_id_name']
col_target_name = config['col_target_name']
dropped_ids = config['dropped_ids']

col_id_name = 'Id'
col_target_name = 'SalePrice'
dropped_ids = [524, 1299]

Xs = load_x(features, dropped_ids)
X_train_all = Xs['train']
X_test = Xs['test']
print_exit(X_test.isnull().sum())
y_train_all = load_y(col_id_name, col_target_name, dropped_ids)

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=6))
model.add(Dense(units=10, activation='softmax'))
model.compile(
    loss= MeanSquaredError(),
    optimizer='sgd',
    metrics=[MeanSquaredError()]
)
model.fit(X_train_all.values, y_train_all.values, epochs=5, batch_size=32)
y_pred_logarithmic = model.predict(X_test.values)
pred = np.expm1(y_pred_logarithmic)
score = r2_score(X_test, pred)
print(score)
