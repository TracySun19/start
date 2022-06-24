
import numpy as np
import torch
from sklearn.datasets import load_iris, load_digits
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


def data_make():
    iris_data = load_iris()
    data, label = iris_data.data, iris_data.target
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3)
    # print(iris_data.data, iris_data.target)
    train_data = lgb.Dataset(data=x_train, label=y_train)
    test_data = lgb.Dataset(data=x_test, label=y_test)

    param = {'num_leaves':31, 'num_trees':100, 'objective':'regression'}
    param['metric'] = 'rmse'
    num_round = 10

    bts = lgb.train(params=param, train_set=train_data, num_boost_round=num_round, valid_sets=[test_data])
    bts.save_model('model.txt')

    y_pred = bts.predict(x_test)
    acc = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'RMSE is {acc:.4f} .')

    # lgb.cv(param, train_data, num_round, nfold=5,early_stopping_rounds=10)

    bts_m = lgb.Booster(model_file='model.txt')


if __name__ == '__main__':
    data_make()