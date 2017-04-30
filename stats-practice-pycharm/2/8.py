import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
import scipy

number_samples = 2000
number_elements_in_samples = 100


def loss_function_quadratic(estimate_realisations, tetta):
    return ((estimate_realisations - tetta) ** 2).mean()


def loss_function_absolute(estimate_realisations, tetta):
    return np.abs(estimate_realisations - tetta).mean()


# (loss_function, label)
loss_functions = [(loss_function_quadratic, '$g(x, y) = (x - y)^2$'),
                  (loss_function_absolute, '$g(x, y) = |x - y|$')]


def get_risk(tetta, distribution, estimator, loss_function):
    """
    возвращает значение функции риска, 
    полученное усреднением значений функции потерь (loss_function) 
    на реализациях оценки (получаются с помощью estimator) на 2000 выборках, 
    сгенерированных из распределения distribution
    """

    samples = distribution.rvs(size=(number_samples, number_elements_in_samples))
    estimate_realisations = estimator(samples)
    loss = loss_function(estimate_realisations, tetta)
    return loss


def draw_risk(grid_for_tetta, estimators, distribution_creator):
    for loss_function, loss_function_label in loss_functions:
        ymax = 100
        for estimator, estimator_label in estimators:
            risks = [get_risk(tetta, distribution_creator(tetta), estimator, loss_function) for tetta in grid_for_tetta]
            plt.plot(grid_for_tetta, risks, label=estimator_label)
            ymax = min(ymax, max(risks))
        plt.grid(ls=':')
        plt.xlabel('$\\theta$', fontsize=16)
        plt.ylabel('$\\widehat{R}\\left(\\theta^*, \\theta\\right)$', fontsize=16)
        plt.legend(fontsize=14)
        plt.title(loss_function_label, fontsize=16)
        plt.ylim((0, ymax * 10))
        plt.show()


def part1():
    def uniform_estimator1(samples):
        return samples.mean(axis=1) * 2

    def uniform_estimator2(samples):
        n = samples.shape[1]
        return samples.min(axis=1) * (n + 1)

    def uniform_estimator3(samples):
        return samples.min(axis=1) + samples.max(axis=1)

    def uniform_estimator4(samples):
        n = samples.shape[1]
        return samples.max(axis=1) * (n + 1) / n

    # (estimator, label)
    uniform_estimators = [(uniform_estimator1, '$2\\overline{X}$'),
                          (uniform_estimator2, '$(n+1)X_{(1)}$'),
                          (uniform_estimator3, '$X_{(n)} + X_{(1)}$'),
                          (uniform_estimator4, '$\\frac{n+1}{n}X_{(n)}$')]
    grid_for_tetta = np.arange(0.01, 5 + 0.01, 0.01)
    draw_risk(grid_for_tetta, uniform_estimators, lambda tetta: sps.uniform(scale=tetta))


def part2():
    def create_exponential_estimator(k):
        k_factorial = scipy.special.gamma(k)

        def exponential_estimator(samples):
            return (k_factorial / (samples ** k).mean(axis=1)) ** 1 / k

        return exponential_estimator

    exponential_estimators = [(create_exponential_estimator(k), '$k = {}$'.format(k)) for k in range(1, 5 + 1)]
    grid_for_tetta = np.arange(0.01, 5 + 0.01, 0.01)
    draw_risk(grid_for_tetta, exponential_estimators, lambda tetta: sps.expon(scale=1 / tetta))
