import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
from math import sqrt, pi

α = 0.95
n = 100


def draw_confidence_interval(left_estimator,  # левая граница интервала
                             right_estimator,  # правая граница интервала
                             true_theta,  # если задана, то рисуется график оценки
                             sample,
                             draw_sample=True,  # рисовать ли точки выборки
                             ylim=(None, None),  # ограничение по оси y
                             title=None):
    xs = np.arange(1, n + 1)
    left = [left_estimator(sample[:i]) for i in xs]
    right = [right_estimator(sample[:i]) for i in xs]

    plt.figure(figsize=(15, 5))
    if draw_sample:
        plt.scatter(xs, sample, alpha=0.5, label='sample')
    plt.fill_between(xs, left, right, alpha=0.15, label='confidence interval')
    plt.plot(xs, [true_theta] * n, color='red', label='true $\\theta$')
    plt.xlim((1, n))
    if ylim != (None, None):
        plt.ylim(ylim)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()


z1 = sps.norm.ppf((1 - α) / 2)
z2 = sps.norm.ppf((1 + α) / 2)
draw_confidence_interval(lambda sample: sample.mean() + z1 / sqrt(len(sample)),
                         lambda sample: sample.mean() + z2 / sqrt(len(sample)),
                         true_theta=0,
                         sample=sps.norm.rvs(size=n),
                         title='Выборка из $N(0,1)$ в модели $N(\\theta,1)$')

draw_confidence_interval(lambda sample: sample.max(),
                         lambda sample: sample.max() / (1 - α) ** (1 / len(sample)),
                         true_theta=1,
                         sample=sps.uniform.rvs(size=n),
                         ylim=(0, 1.5),
                         title='Выборка из $U[0,1]$ в модели $U[0,\\theta]$')

draw_confidence_interval(lambda sample: (sqrt(2 / len(sample)) * z1 + 2) / sample.mean(),
                         lambda sample: (sqrt(2 / len(sample)) * z2 + 2) / sample.mean(),
                         true_theta=3,
                         sample=sps.gamma(a=2, scale=1 / 3).rvs(size=n),
                         draw_sample=False,
                         title='Выборка из $Г(3,2)$ в модели $Г(\\theta,2)$',
                         ylim=(-1, 7))

draw_confidence_interval(lambda sample: (pi * z1) / (2 * sqrt(len(sample))),
                         lambda sample: (pi * z2) / (2 * sqrt(len(sample))),
                         true_theta=0,
                         sample=sps.cauchy.rvs(size=n),
                         ylim=(-5, 5),
                         title='Выборка из $Cauchy$ в модели $Cauchy(\\theta)$')

draw_confidence_interval(lambda sample: sample.mean() + z1 / sqrt(len(sample)),
                         lambda sample: sample.mean() + z2 / sqrt(len(sample)),
                         true_theta=0,
                         sample=sps.cauchy.rvs(size=n),
                         ylim=(-7, 7),
                         title='Выборка из $Cauchy$ в модели $N(0,1)$')
