import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps


def part1():
    grid = np.linspace(0, 5, 1000)
    plt.figure(figsize=(15, 4))
    for λ, color in [(1, 'red'), (3, 'green'), (10, 'blue')]:
        plt.plot(grid, sps.expon(scale=1 / λ).pdf(grid), lw=3, color=color, label='$\\lambda={}$'.format(λ))
        plt.legend(fontsize=16)
        plt.ylim((0, 2))
        plt.grid(ls=':')
    plt.show()


def draw(function, xi_label):
    grid = np.linspace(0, 5, 1000)
    plt.figure(figsize=(15, 5))
    plt.plot(grid, [function(x) for x in grid], lw=3, label='${}$'.format(xi_label))
    for λ, color in [(1, 'red'), (3, 'green'), (10, 'cyan')]:
        distribution = sps.expon(scale=1 / λ)
        # [0, 5) = [0, 1) ⨆ ... ⨆ [4, 5)
        # D_i := [i, i + 1)
        for i in range(5):  # события из сигма-алгебры
            # E(ξI_{D_i})
            expect = distribution.expect(function, lb=i, ub=i + 1)

            # P(D_i)
            p_d_i = distribution.expect(lambda x: 1, lb=i, ub=i + 1)
            # вот так не хватает точности:
            # p_d_i = distribution.cdf(i + 1) - distribution.cdf(i)

            plt.hlines(expect / p_d_i, i, i + 1, color=color, lw=3, label=('$\\mathsf{{E}}({}|\\mathcal{{G}})$ при $\\lambda = {}$'.format(xi_label, λ) if i == 0 else ''))
        plt.xlabel('$\\Omega$', fontsize=20)
        plt.legend(fontsize=16)
        plt.grid(ls=':')
    plt.show()


draw(lambda x: x, '\\xi')
draw(lambda x: x * x, '\\xi^2')
