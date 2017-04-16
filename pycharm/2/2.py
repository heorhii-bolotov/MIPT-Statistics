import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps


# cell 0
def check_consistent(distribution, estimator):
    number_samples = 200
    number_elements_in_sample = 300
    samples = distribution.rvs(size=number_elements_in_sample * number_samples).reshape(number_samples, number_elements_in_sample)
    estimates = estimator(samples)
    for estimate in estimates:
        plt.plot(np.arange(1, number_elements_in_sample + 1), estimate, color='red', alpha=0.2)
    plt.show()


def cell2():
    def norm_estimator(samples):
        n = samples.shape[1]
        return samples.cumsum(axis=1) / np.arange(1, n + 1)

    check_consistent(sps.norm, norm_estimator)


def cell3():
    def uniform_estimator(samples):
        return np.maximum.accumulate(samples, axis=1)

    check_consistent(sps.uniform, uniform_estimator)
