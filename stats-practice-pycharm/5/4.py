import numpy as np
import scipy.stats as sps
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import math
import tabulate
from IPython.display import HTML, display


class LinearRegression:
    def __init__(self):
        super()

    def fit(self, X, Y, alpha=0.95):
        """ Обучение модели. Предполагается модель Y = X * theta + epsilon, 
            где X --- регрессор, Y --- отклик,
            а epsilon имеет нормальное распределение с параметрами (0, sigma^2 * I_n).
            alpha --- уровень доверия для доверительного интервала.
        """

        self.n, self.k = X.shape

        self.inv_of_xt_dot_x = np.linalg.inv(X.T @ X)
        self.theta = self.inv_of_xt_dot_x @ X.T @ Y  # МНК-оценка
        self.sigma_sq = ((Y - X @ self.theta) ** 2).sum() / (self.n - self.k)  # несмещённая оценка для sigma^2
        self.conf_int = np.array([
            self.theta - np.sqrt(self.inv_of_xt_dot_x.diagonal() * self.sigma_sq) * sps.t(self.n - self.k).ppf((1 + alpha) / 2),
            self.theta - np.sqrt(self.inv_of_xt_dot_x.diagonal() * self.sigma_sq) * sps.t(self.n - self.k).ppf((1 - alpha) / 2)
        ]).T  # доверительные интервалы для коэффициентов (матрица размера k x 2)

        return self

    def summary(self):
        print('Linear regression on %d features and %d examples' % (self.k, self.n))
        print('Sigma: %.6f' % self.sigma_sq)
        print('\t\tLower\t\tEstimation\tUpper')
        for j in range(self.k):
            print('theta_%d:\t%.6f\t%.6f\t%.6f' % (j + 1, self.conf_int[j, 0],
                                                   self.theta[j], self.conf_int[j, 1]))

    def predict(self, X):
        """ Возвращает предсказание отклика на новых объектах X. """

        Y_pred = X @ self.theta
        return Y_pred


table = [['n', '', 'доверительный интервал для a', '', 'доверительный интервал для $\sigma^2$']]
sample = sps.norm.rvs(size=50)
for n in [5, 20, 50]:
    Y = sample[:n].T
    X = np.ones((n, 1))
    model = LinearRegression().fit(X, Y, alpha=1 - 0.05 / 2)
    доверительный_интервал_для_a = model.conf_int[0]
    доверительный_интервал_для_сигма2 = [0, (n - 1) * model.sigma_sq / sps.chi2(n - 1).ppf(math.sqrt(0.95))]
    line = (n, *доверительный_интервал_для_a, *доверительный_интервал_для_сигма2)
    table.append(line)
display(HTML(tabulate.tabulate(table, tablefmt='html')))
