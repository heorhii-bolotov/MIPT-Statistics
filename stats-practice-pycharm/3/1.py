import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
from math import *


def part1():
    grid = np.linspace(0, 5, 1000)
    plt.figure(figsize=(15, 4))
    for λ, color in [(1, 'red'), (3, 'green'), (10, 'blue')]:
        plt.plot(grid, sps.expon(scale=1 / λ).pdf(grid), lw=3, color=color, label='$\\lambda={}$'.format(λ))
        plt.legend(fontsize=16)
        plt.ylim((0, 2))
        plt.grid(ls=':')
    plt.show()


def draw(xi_label, function, integral):
    grid = np.linspace(0, 5, 1000)
    plt.figure(figsize=(15, 5))
    plt.plot(grid, [function(x) for x in grid], lw=3, label='${}$'.format(xi_label))
    for λ, color in [(1, 'red'), (3, 'green'), (10, 'cyan')]:
        # [0, 5) = [0, 1) ⨆ ... ⨆ [4, 5)
        # D_i := [i, i + 1)
        for i in range(5):  # события из сигма-алгебры
            # Для графика ξ: E(ξ * I_{D_i}) = int λ * x * exp(-λx) = -exp(-λx)(λx+1)/λ from i to i+1
            # Для графика ξ^2: E(ξ^2 * I_{D_i}) = int λ * x^2 * exp(-λx) = -exp(-λx)(λ^2*x^2 + 2λx + 2) / λ^2 from i to i+1
            expect = integral(λ, i + 1) - integral(λ, i)

            # P(D_i) = P([i, i + 1)) = (1 - exp(-λ(i+1))) - (1 - exp(-λi))
            p_d_i = exp(-λ * i) - exp(-λ * (i + 1))

            plt.hlines(expect / p_d_i, i, i + 1, color=color, lw=3, label=('$\\mathsf{{E}}({}|\\mathcal{{G}})$ при $\\lambda = {}$'.format(xi_label, λ) if i == 0 else ''))
        plt.xlabel('$\\Omega$', fontsize=20)
        plt.legend(fontsize=16)
        plt.grid(ls=':')
    plt.show()


draw('\\xi', lambda x: x, lambda λ, x: -exp(-λ * x) * (λ * x + 1) / λ)
draw('\\xi^2', lambda x: x * x, lambda λ, x: -exp(-λ * x) * (λ ** 2 * x ** 2 + 2 * λ * x + 2) / λ ** 2)
