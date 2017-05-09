import numpy as np
import scipy.stats as sps
from statsmodels.distributions.empirical_distribution import ECDF
from math import sqrt


def get_supremum(sample, cdf):
    """По выборке строит эмпирическую функцию распределения ecdf, считает точное значение статистики sup|ecdf - cdf|"""
    eps = 10 ** -7
    ecdf = ECDF(sample)
    return max([np.abs(ecdf(sample + offset) - cdf(sample)).max() for offset in (-eps, 0)])


def normal_summary(sample, name=None):
    α = 0.95

    n = len(sample)
    mean = sample.mean()
    std = sample.std()
    distribution = sps.norm(loc=mean, scale=std)

    if name is not None:
        print('\tSummary of {}:'.format(name))
    print('size: %d' % n)
    print('sample mean: %.2f' % mean)
    print('sample median: %.2f' % np.median(sample))
    print('sample std: %.2f' % std)  # стандартное отклонение == корень из дисперсии
    print('0.95 confidence interval: (%.2f, %.2f)' %
          (distribution.ppf((1 - α) / 2),
           distribution.ppf((1 + α) / 2)))
    # значение статистики из теоремы Колмогорова-Смирнова,
    # взяв в качестве F функцию распределения нормального
    # распределения с оценёнными выше параметрами
    print('KS-stat: %.3f' % get_supremum(sample, distribution.cdf))
    print()


normal_summary(sps.norm.rvs(size=10), 'N(0, 1)')
normal_summary(sps.norm.rvs(size=100), 'N(0, 1)')
normal_summary(sps.norm.rvs(size=178), 'N(0, 1)')
normal_summary(sps.norm.rvs(size=1000), 'N(0, 1)')

normal_summary(sps.norm(loc=0).rvs(size=100), 'N(0, 1)')
normal_summary(sps.norm(loc=1).rvs(size=100), 'N(1, 1)')
normal_summary(sps.norm(loc=10).rvs(size=100), 'N(10, 1)')
normal_summary(sps.norm(loc=100).rvs(size=100), 'N(100, 1)')
normal_summary(sps.norm(loc=-100).rvs(size=100), 'N(-100, 1)')

normal_summary(sps.norm(scale=1).rvs(size=100), 'N(0, 1)')
normal_summary(sps.norm(scale=2).rvs(size=100), 'N(0, 4)')
normal_summary(sps.norm(scale=10).rvs(size=100), 'N(0, 100)')
normal_summary(sps.norm(scale=100).rvs(size=100), 'N(0, 10000)')

normal_summary(sps.norm(loc=13, scale=1).rvs(size=178), 'N(13, 1)')
normal_summary(sps.norm(loc=19.5, scale=3.33).rvs(size=178), 'N(19.5, 3.33)')
normal_summary(sps.norm(loc=0.35, scale=0.1).rvs(size=178), 'N(0.35, 0.1)')

normal_summary(sps.uniform.rvs(size=10), 'U(0, 1)')
normal_summary(sps.expon.rvs(size=10), 'Exp(1)')
normal_summary(sps.uniform.rvs(size=100), 'U(0, 1)')
normal_summary(sps.expon.rvs(size=100), 'Exp(1)')
normal_summary(sps.uniform.rvs(size=1000), 'U(0, 1)')
normal_summary(sps.expon.rvs(size=1000), 'Exp(1)')

data = np.loadtxt('wine.data', delimiter=',')
normal_summary(data[:, 1], 'alcohol')
normal_summary(data[:, 4], 'alcalinity of ash')
normal_summary(data[:, 8], 'nonflavanoid phenols')
