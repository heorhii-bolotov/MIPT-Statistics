import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps


def cool_argmax(array):
    return np.unravel_index(np.argmax(array), array.shape)


# находит ОМП по сетке
def best_likelihood(grid, sample):
    # плотности распределения Вейбулла в точках grid
    logpdf = sps.weibull_min(c=grid[0], scale=grid[1]).logpdf
    # функции правдоподобия в точках grid
    logpdfs = logpdf(sample[:, np.newaxis, np.newaxis])
    likelihood = logpdfs.sum(axis=0)
    # индекс сетки, в котором функция правдоподобия максимальна
    argmax = cool_argmax(likelihood)
    # (k, λ)
    return grid[0][argmax], grid[1][argmax]


def part1():
    global sample, k, λ
    # Среднесуточная скорость ветра за 2012 год
    # http://www.atlas-yakutia.ru/weather/wind/climate_russia-III_wind.html
    lines = open('7.txt', 'r').readlines()
    sample = np.array(list(map(float, lines)))
    # начальная сетка --- [1..100, 1..100] с шагом 1
    from_k, to_k = 1, 100
    from_λ, to_λ = 1, 100
    # объявляем переменые вне цикла, чтобы после цикла они сохранились
    k, λ = 0, 0
    # меняем шаг сетки с 1 до 1e-6
    for step in np.logspace(0, -6, 7):
        k, λ = best_likelihood(np.mgrid[from_k:to_k:step, from_λ:to_λ:step], sample)

        print('step:           {}'.format(step))
        print('grid, k:        {:.10g} .. {:.10g}'.format(from_k, to_k))
        print('grid, λ:        {:.10g} .. {:.10g}'.format(from_λ, to_λ))
        print('k:              {:.10g}'.format(k))
        print('λ:              {:.10g}'.format(λ))
        print()

        # строим новую сетку меньшего размера (с меньшим шагом)
        from_k = max(0.1, k - (step * 10))
        to_k = k + (step * 10)
        from_λ = max(0.1, λ - (step * 10))
        to_λ = λ + (step * 10)


def part2():
    grid = np.linspace(0, 10, 500)
    # гистограмма
    plt.hist(sample, 20, range=(grid.min(), grid.max()), normed=True, label='histogram')
    # истинная плотность
    plt.plot(grid, sps.weibull_min(c=k, scale=λ).pdf(grid), color='red', linewidth=2, alpha=0.3, label='true pdf')
    plt.legend()
    plt.show()
