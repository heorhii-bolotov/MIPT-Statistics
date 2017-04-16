import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps


# cell 1
def check_unbiased(distribution, estimators):
    number_samples = 500
    for number_elements_in_sample in [10, 100, 500]:
        # Для каждой оценки:
        samples = distribution.rvs(size=number_elements_in_sample * number_samples).reshape(number_samples, number_elements_in_sample)
        for i, (estimator, estimate_label, estimate_color) in enumerate(estimators):
            estimate = estimator(samples)
            plt.scatter(estimate, np.zeros_like(estimate) + i, alpha=0.1, s=100, color=estimate_color, label=estimate_label)
            plt.scatter(estimate.mean(), i, marker='*', s=200, color='w', edgecolors='black')

        # Для всего графика:
        plt.vlines(1, -0.5, len(estimators) - 0.5, color='r')
        plt.title('sample size = %d' % number_elements_in_sample)
        plt.yticks([])
        plt.legend()
        plt.show()


def cell2():
    def uniform_estimator1(samples):
        return samples.max(axis=1)

    def uniform_estimator2(samples):
        n = samples.shape[1]
        return samples.max(axis=1) * (n + 1) / n

    def uniform_estimator3(samples):
        return samples.mean(axis=1) * 2

    uniform_estimators = [(uniform_estimator1, '$X_{(n)}$', 'red'),
                          (uniform_estimator2, '$\\frac{n+1}{n}X_{(n)}$', 'green'),
                          (uniform_estimator3, '$2\overline{X}$', 'blue')]
    check_unbiased(sps.uniform, uniform_estimators)


def cell3():
    def norm_estimator1(samples):
        return samples.std(axis=1)

    def norm_estimator2(samples):
        n = samples.shape[1]
        return samples.std(axis=1) * n / (n - 1)

    norm_estimators = [(norm_estimator1, '$S^2$', 'red'),
                       (norm_estimator2, '$\\frac{n}{n-1}S^2$', 'green')]
    check_unbiased(sps.norm, norm_estimators)
