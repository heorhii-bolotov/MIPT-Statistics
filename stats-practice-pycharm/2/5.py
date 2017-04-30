import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps


def draw_likelihood(density_function, grid, samples, label):
    ''' density_function --- функция, считающая плотность (обычную или дискретную)
        grid --- сетка для построения графика
        samples --- три выборки
        label --- latex-код параметрической модели
    '''

    plt.figure(figsize=(18, 5))
    for i, sample in enumerate(samples):
        sample = np.array(sample)[np.newaxis, :]
        # значение функции правдоподобия
        likelihood = density_function(sample).prod(axis=1)

        plt.subplot(1, 3, i + 1)
        plt.plot(grid, likelihood)
        plt.xlabel('$\\theta$', fontsize=16)
        plt.grid(ls=':')
        plt.title(label + ', sample=' + str(sample), fontsize=16)
    plt.show()


def cell1():
    grid = np.linspace(-5, 5, 1000).reshape((-1, 1))
    draw_likelihood(sps.norm(loc=grid).pdf, grid, [[-1, 1], [-5, 5], [-1, 5]], '$\\mathcal{N}(\\theta, 1)$')
    draw_likelihood(sps.expon(loc=grid).pdf, grid, [[1, 2], [0.1, 1], [1, 10]], '$Exp(\\theta)$')
    draw_likelihood(sps.uniform(scale=grid).pdf, grid, [[0.2, 0.8], [0.5, 1], [0.5, 1.3]], '$U[0, \\theta]$')
    draw_likelihood(sps.binom(n=5, p=grid).pmf, grid, [[0, 1], [5, 5], [0, 5]], '$Bin(5, \\theta)$')
    draw_likelihood(sps.poisson(mu=grid).pmf, grid, [[0, 1], [0, 10], [5, 10]], '$Pois(\\theta)$')
    draw_likelihood(sps.cauchy(loc=grid).pdf, grid, [[-0.5, 0.5], [-2, 2], [-4, 0, 4]], '$Сauchy(\\theta)$')


def cell2():
    sample = sps.norm.rvs(size=10 ** 5)
    loglikelihood = sps.norm.logpdf(sample).sum()
    print(loglikelihood)


grid = np.linspace(-5, 5, 1000).reshape((-1, 1))
draw_likelihood(sps.norm(loc=grid).pdf, grid, [[-1, 1], [-5, 5], [-1, 5]], '$\\mathcal{N}(\\theta, 1)$')
draw_likelihood(sps.expon(loc=grid).pdf, grid, [[1, 2], [0.1, 1], [1, 10]], '$Exp(\\theta)$')
draw_likelihood(sps.uniform(scale=grid).pdf, grid, [[0.2, 0.8], [0.5, 1], [0.5, 1.3]], '$U[0, \\theta]$')
draw_likelihood(sps.binom(n=5, p=grid).pmf, grid, [[0, 1], [5, 5], [0, 5]], '$Bin(5, \\theta)$')
draw_likelihood(sps.poisson(mu=grid).pmf, grid, [[0, 1], [0, 10], [5, 10]], '$Pois(\\theta)$')
draw_likelihood(sps.cauchy(loc=grid).pdf, grid, [[-0.5, 0.5], [-2, 2], [-4, 0, 4]], '$Сauchy(\\theta)$')