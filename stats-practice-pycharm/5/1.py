import numpy as np
import scipy.stats as sps
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from math import sqrt


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


# считываем данные из файла
data = pd.read_csv('ice_cream.txt', delimiter='\t')
ic = data['IC'].values
temperature = data['temp'].values
temperature = (temperature - 32) / 1.8  # переводим из Фаренгейтов в Цельсии

# модель ic = θ1 + θ2*t
regressor = np.column_stack((np.ones_like(temperature), temperature))
linear_regression = LinearRegression().fit(regressor, ic)
linear_regression.summary()

# график для модели ic = θ1 + θ2*t
# plt.figure(figsize=(13, 10))
# plt.ylabel('Потребление мороженого')
# plt.xlabel('Температура')
# temperature_grid = np.linspace(temperature.min() - 5, temperature.max() + 5)
# temperature_grid_predict = linear_regression.predict(np.column_stack((np.ones_like(temperature_grid), temperature_grid)))
# plt.plot(temperature_grid, temperature_grid_predict)
# plt.scatter(temperature, ic, color='red')
# plt.title(r'Модель $ic = θ1 + θ2*t$')
# plt.show()

# модель ic = θ1 + θ2*t + θ3*y1 + θ4*y2
years = data['Year'].values
regressor = np.column_stack((np.ones_like(temperature), temperature, years == 1, years == 2))
linear_regression = LinearRegression().fit(regressor, ic)
linear_regression.summary()

# график для модели ic = θ1 + θ2*t + θ3*y1 + θ4*y2
# plt.figure(figsize=(13, 10))
# plt.ylabel('Потребление мороженого')
# plt.xlabel('Температура')
# for year in range(3):
#     temperature_grid = np.linspace(temperature.min() - 5, temperature.max() + 5)
#     temperature_grid_predict = linear_regression.predict(np.column_stack(
#         (np.ones_like(temperature_grid),
#          temperature_grid,
#          np.full_like(temperature_grid, year == 1),
#          np.full_like(temperature_grid, year == 2))))
#     plt.plot(temperature_grid, temperature_grid_predict,
#              label='Модель ic = θ1 + θ2*t + θ3*y1 + θ4*y2, y1={}, y2={}'.format(year == 1, year == 2))
#     indixes = years == year
#     plt.scatter(temperature[indixes], ic[indixes], label='точки выборки, год {}'.format(year))
# plt.title(r'Модель $ic = θ1 + θ2*t + θ3*y1 + θ4*y2$')
# plt.legend()
# plt.show()

# модель ic = θ1 + θ2*t, отдельно для каждого года
# plt.figure(figsize=(13, 10))
# plt.ylabel('Потребление мороженого')
# plt.xlabel('Температура')
# for year in range(3):
#     indixes = years == year
#     temperature_year = temperature[indixes]
#     ic_year = ic[indixes]
#
#     regressor = np.column_stack((np.ones_like(temperature_year), temperature_year))
#     linear_regression = LinearRegression().fit(regressor, ic_year)
#     temperature_grid = np.linspace(temperature.min() - 5, temperature.max() + 5)
#     temperature_grid_predict = linear_regression.predict(np.column_stack((np.ones_like(temperature_grid), temperature_grid)))
#     plt.plot(temperature_grid, temperature_grid_predict, label='Модель ic = θ1 + θ2*t, год {}'.format(year))
#     plt.scatter(temperature_year, ic_year, label='точки выборки, год {}'.format(year))
# plt.title(r'Модель $ic = θ1 + θ2*t$, отдельно для каждого года')
# plt.legend()
# plt.show()

# модель ic = θ1 + θ2*t + θ3*y1 + θ4*y2 + θ5*price + θ6*income + θ7*lag_temp
price = data['price'].values
income = data['income'].values
lag_temp = data['Lag-temp'].values
regressor = np.column_stack(
    (np.ones_like(temperature),
     temperature,
     years == 1,
     years == 2,
     price,
     income,
     lag_temp))
linear_regression = LinearRegression().fit(regressor, ic)
linear_regression.summary()

# модель ic = θ1 + θ2*t + θ3*t**2 + θ3*t**3
regressor = np.column_stack(
    (np.ones_like(temperature),
     temperature,
     temperature ** 2,
     temperature ** 3))
linear_regression = LinearRegression().fit(regressor, ic)
linear_regression.summary()

# график для модели ic = θ1 + θ2*t + θ3*t**2 + θ3*t**3
# plt.figure(figsize=(13, 10))
# plt.ylabel('Потребление мороженого')
# plt.xlabel('Температура')
# temperature_grid = np.linspace(temperature.min() - 5, temperature.max() + 5)
# temperature_grid_predict = linear_regression.predict(np.column_stack(
#     (np.ones_like(temperature_grid),
#      temperature_grid,
#      temperature_grid ** 2,
#      temperature_grid ** 3)))
# plt.plot(temperature_grid, temperature_grid_predict)
# plt.scatter(temperature, ic, color='red')
# plt.title(r'Модель $ic = θ1 + θ2*t + θ3*t^2 + θ3*t^3$')
# plt.show()


# собственные значения
eigvals = scipy.linalg.eigvals(linear_regression.inv_of_xt_dot_x)
print(eigvals)

# индекс обусловленности
# scipy.linalg.eigvals возвращает комплексные числа, однако мнимая часть у них равна нулю
condition_number = sqrt(eigvals.max().real / eigvals.min().real)
print(condition_number)
