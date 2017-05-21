import numpy as np
import scipy.stats as sps
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston


def best_features(X_train, X_test, Y_train, Y_test):
    mses = []  # сюда записывайте значения MSE
    k = X_train.shape[1]

    for j in range(1, 2 ** k):  # номер набора признаков
        mask = np.array([j & (1 << s) for s in range(k)], dtype=bool)
        features_numbers = np.arange(k)[mask]  # набор признаков

        model = LinearRegression().fit(X_train[:, features_numbers], Y_train)
        Y_test_predict = model.predict(X_test[:, features_numbers])
        mse = mean_squared_error(Y_test, Y_test_predict)  # MSE для данного набора признаков
        mses.append(mse)

    # Печать 10 лучших наборов
    print('mse\t features')
    mses = np.array(mses)
    best_numbres = np.argsort(mses)[:10]
    for j in best_numbres:
        mask = np.array([j & (1 << s) for s in range(k)], dtype=bool)
        features_numbers = np.arange(k)[mask]
        print('%.3f\t' % mses[j], features_numbers)


# Парусные яхты: остаточное сопротивление на единицу массы смещения в зависимости от различных характеристик яхты
data = np.loadtxt('yacht_hydrodynamics.data')
X = data[:, :-1]
Y = data[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
best_features(X_train, X_test, Y_train, Y_test)

# цены на дома в Бостоне в зависимости от ряда особенностей
X, Y = load_boston(return_X_y=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
best_features(X_train, X_test, Y_train, Y_test)
