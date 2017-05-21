import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression


def part1():
    sample = np.loadtxt('выборка_скорости_ветра_из_задания_2.txt')
    shape, loc, scale = sps.weibull_min.fit(sample, floc=0)
    print(shape, loc, scale)

    kstatisics, pvalue = sps.kstest(sample, sps.weibull_min(shape, scale=scale).cdf)
    print(kstatisics, pvalue)


def part2():
    def check(distribution, estimator, true_distribution_of_T_j_n, distribution_name):
        number_samples = 200
        number_elements_in_sample = 300
        samples = distribution.rvs(size=(number_elements_in_sample, number_samples))
        estimates = estimator(samples)
        # зотим проверить, правда ли, что estimates --- выборка из распределения true_distribution_of_T_j_n
        kstatisics, pvalue = sps.kstest(estimates, true_distribution_of_T_j_n.cdf)
        print(distribution_name, kstatisics, pvalue)

    def create_estimator(distribution_mean):
        def estimator(samples):
            # принимает массив выборок
            # для каждой выборки считает статистику: T = \sqrt{n} * (θ_n - θ),
            # где θ_n = (X_1 + ... + X_n) / n
            # n --- размер выборки (=300)
            n = samples.shape[1]
            estimates = samples.mean(axis=1)
            estimates = math.sqrt(n) * (estimates - distribution_mean)
            return estimates

        return estimator

    def estimator(samples):
        # принимает массив выборок
        # для каждой выборки считает статистику: T = n * (θ - X_(n))
        # где θ=1
        n = samples.shape[1]
        return n * (1 - samples.max(axis=1))

    check(sps.norm, create_estimator(0), sps.norm, 'normal ')
    check(sps.poisson(1), create_estimator(1), sps.norm, 'poisson')
    check(sps.uniform, estimator, sps.expon, 'uniform')


sample = np.loadtxt('выборка_числа_друзей_из_задания_1.csv')
kstatisics, pvalue = sps.kstest(sample, sps.norm(sample.mean(), sample.std()).cdf)
print(kstatisics, pvalue)

омп_для_парето = 1 / np.log(sample + 1).mean()
print(омп_для_парето)
kstatisics, pvalue = sps.kstest(sample + 1, sps.pareto(омп_для_парето).cdf)
print(kstatisics, pvalue)
