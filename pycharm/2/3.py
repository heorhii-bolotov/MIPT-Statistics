from statsmodels.nonparametric.kde import KDEUnivariate
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps


def draw_hist_and_kde(sample, grid, true_pdf):
    # гистограмма
    plt.hist(sample, 20, range=(grid.min(), grid.max()), normed=True, label='histogram')

    # ядерная оценка плотности
    kernel_density = KDEUnivariate(sample)
    kernel_density.fit()
    plt.plot(grid, kernel_density.evaluate(grid), color='green', linewidth=2, label='kde')

    # истинная плотность
    plt.plot(grid, true_pdf(grid), color='red', linewidth=2, alpha=0.3, label='true pdf')

    plt.legend()
    plt.show()


def check_asymptotic_normality(distribution, estimator, true_distribution_of_T_j_n, grid_min, grid_max, distribution_name):
    number_samples = 200
    number_elements_in_sample = 300
    samples = distribution.rvs(size=number_elements_in_sample * number_samples).reshape(number_samples, number_elements_in_sample)
    estimates = estimator(samples)

    plt.title('plots of $T_{{jn}}$ for {} distribution'.format(distribution_name))
    for estimate in estimates:
        plt.plot(np.arange(1, number_elements_in_sample + 1), estimate, color='red', alpha=0.2)
    plt.show()

    plt.title('plots of $T_{{j, {}}}$ for {} distribution'.format(number_elements_in_sample, distribution_name))
    draw_hist_and_kde(estimates[:, -1], np.linspace(grid_min, grid_max, 500), true_distribution_of_T_j_n.pdf)


def task3():
    def create_estimator(distribution_mean):
        def estimator(samples):
            # принимает массив выборок
            # для каждой выборки считает n статистик: T_k = \sqrt{k} * (θ_k - θ),
            # где θ_k = (X_1 + ... + X_k) / k
            n = samples.shape[1]
            estimates = samples.cumsum(axis=1) / np.arange(1, n + 1)
            estimates = np.sqrt(np.arange(1, n + 1)) * (estimates - distribution_mean)
            return estimates

        return estimator

    check_asymptotic_normality(sps.norm, create_estimator(0), sps.norm, -4, 4, 'normal')
    check_asymptotic_normality(sps.poisson(1), create_estimator(1), sps.norm, -4, 4, 'poisson')


def task4():
    def estimator(samples):
        # принимает массив выборок
        # для каждой выборки считает n статистик: T_k = k * (θ - X_(k))
        n = samples.shape[1]
        return np.arange(1, n + 1) * (1 - np.maximum.accumulate(samples, axis=1))

    check_asymptotic_normality(sps.uniform, estimator, sps.expon, -1, 8, 'uniform')
