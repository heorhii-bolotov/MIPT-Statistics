import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps


def part1():
    grid = np.linspace(-7, 7, 500)
    plt.plot(grid, sps.norm.pdf(grid), label='Нормальное')
    plt.plot(grid, sps.cauchy.pdf(grid), label='Коши')
    plt.legend()
    plt.show()


def part2():
    number_elements_in_sample = 10000
    sample = sps.cauchy.rvs(size=number_elements_in_sample)
    averages = [sample[:n].mean() for n in range(1, number_elements_in_sample + 1)]
    medians = [np.median(sample[:n]) for n in range(1, number_elements_in_sample + 1)]

    xs = np.arange(1, number_elements_in_sample + 1)
    plt.plot(xs, averages, label='выборочное среднее')
    plt.plot(xs, medians, label='выборочная медиана')
    plt.legend()
    plt.show()
