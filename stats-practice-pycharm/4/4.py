import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from math import sqrt, pi
from heapq import heappush, heappop

K = 10 ** 5
N = 300


def get_supremum(sample):
    """По выборке строит эмпирическую функцию распределения F*, считает точное значение статистики sup|F* - F|, где F --- функция распределения N(0, 1)"""
    eps = 10 ** -7
    ecdf = ECDF(sample)
    return max([np.abs(ecdf(sample + offset) - sps.norm.cdf(sample)).max() for offset in (-eps, 0)])


def cumavg(X):
    """
    По матрице (x_i_j) строит последовательность средних частичных сумм каждой строки
    result_i_j == X[i, :j+1].mean()
    """
    return X.cumsum(axis=1) / np.arange(1, len(X[0]) + 1)


def cumstd(X):
    """
    По матрице (x_i_j) строит последовательность стандартных отклонений (S^2) каждой строки
    result_i_j == X[i, :j+1].std()
    """
    return np.sqrt(cumavg(X ** 2) - cumavg(X) ** 2)


def cummedian_line_slow(a):
    """
    По строке (a_i) строит последовательность медиан
    result_i == a[:i+1].median()
    """
    return np.array([np.median(a[:i + 1]) for i in range(len(a))])


def cummedian_line(a):
    """
    По строке (a_i) строит последовательность медиан
    result_i == a[:i+1].median()

    http://stackoverflow.com/a/10657732
    Аж O(nlog(n))!!!
    """

    class minheap(list):
        def heappush(self, value):
            heappush(self, value)

        def heappop(self):
            return heappop(self)

        def heappeek(self):
            return self[0]

    class maxheap(list):
        def heappush(self, value):
            heappush(self, -value)

        def heappop(self):
            return -heappop(self)

        def heappeek(self):
            return -self[0]

    assert len(a) >= 2
    halfmin = maxheap()
    halfmax = minheap()
    halfmin.heappush(min(a[:2]))
    halfmax.heappush(max(a[:2]))
    medians = [a[0], (a[0] + a[1]) / 2]
    a = a[2:]

    for ai in a:
        (halfmin if ai <= halfmax.heappeek() else halfmax).heappush(ai)
        if len(halfmin) - len(halfmax) > 1:
            halfmax.heappush(halfmin.heappop())
        if len(halfmax) - len(halfmin) > 1:
            halfmin.heappush(halfmax.heappop())
        assert abs(len(halfmin) - len(halfmax)) <= 1
        medians.append((halfmin.heappeek() + halfmax.heappeek()) / 2 if (len(halfmin) + len(halfmax)) % 2 == 0 else (halfmin if len(halfmin) > len(halfmax) else halfmax).heappeek())
    return np.array(medians)


def cummedian_matrix(X):
    """
    По матрице (x_i_j) строит последовательность медиан каждой строки
    result_i_j == X[i, :j+1].median()
    """
    return np.array([cummedian_line(line) for line in X])


def draw(rvs, t_estimators, title, ylim=None):
    samples = rvs(size=(K, N))
    for t_estimator, t_estimator_label in t_estimators:
        ts = t_estimator(samples)
        supremums = [get_supremum(ts[:, n]) for n in range(N)]
        plt.plot(range(N), supremums, label=t_estimator_label)
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
    plt.grid(ls=':')
    plt.legend()
    plt.show()


def part1():
    t_estimators = [
        (lambda samples: np.sqrt(np.arange(1, N + 1)) * cumavg(samples), '$\sqrt{n}\cdot\overline{X}$'),
        (lambda samples: np.sqrt(np.arange(1, N + 1)) * cumavg(samples) / cumstd(samples), '$\sqrt{n}\cdot\overline{X}/\sqrt{S^2}$')
    ]
    draw(sps.norm.rvs, t_estimators, 'Выборка из $N(0, 1)$', (0, 0.05))


def part2():
    t_estimators = [
        (lambda samples: np.sqrt(np.arange(1, N + 1)) * (cumavg(samples) - 0.5) * sqrt(2), '$\sqrt{n} \\frac{\overline{X} - p}{\sqrt{p(1-p)}}$'),
        (lambda samples: np.sqrt(np.arange(1, N + 1)) * (cumavg(samples) - 0.5) / cumstd(samples), '$\sqrt{n} \\frac{\overline{X} - p}{\sqrt{S^2}}$')
    ]
    draw(sps.bernoulli(0.5).rvs, t_estimators, 'Выборка из $Bern(0.5)$')


def part3():
    t_estimators = [(lambda samples: np.sqrt(np.arange(1, N + 1)) * cummedian_matrix(samples) / (pi / 2), '$\sqrt{n} \\frac{\widehat\mu}{\pi/2}$')]
    draw(sps.cauchy.rvs, t_estimators, 'Выборка из $Cauchy$', (0, 0.05))


part3()
