import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt


def part1():
    plt.figure(figsize=(15, 7))
    grid = np.linspace(0, 1, 100)
    parameters = [
        [7, 7],
        [0.7, 0.7],
        [3, 1],
        [4, 3],
        [1, 1]
    ]
    for a, b in parameters:
        plt.plot(grid, sps.beta(a=a, b=b).pdf(grid), label='a={}, b={}'.format(a, b))
    plt.legend()
    plt.grid(ls=':')
    plt.xlim((0, 1))
    plt.show()


def draw_posteriori(grid, distr_class, post_params, xlim=None):
    ''' Рисует серию графиков апостериорных плотностей.
        grid --- сетка для построения графика
        distr_class --- класс распределений из scipy.stats
        post_params --- параметры апостериорных распределений 
                        shape=(размер выборки, кол-во параметров)
    '''

    size = post_params.shape[0] - 1

    plt.figure(figsize=(12, 7))
    for n in range(size + 1):
        plt.plot(grid,
                 distr_class(post_params[n]).pdf(grid) if np.isscalar(post_params[n])
                 else distr_class(*post_params[n]).pdf(grid),
                 label='n={}: {}'.format(n, post_params[n]),
                 lw=2.5,
                 color=(1 - n / size, n / size, 0))
    plt.grid(ls=':')
    plt.legend()
    plt.xlim(xlim)
    plt.show()


def draw_estimations(ml, distr_class, post_params, confint=True, ylim=None, xlim=None, title=''):
    ''' Рисует графики байесовской оценки (м.о. и дов. инт.) и ОМП.
        ml --- Оценка максимального правдоподобия для 1 <= n <= len(sample)
        distr_class --- класс распределений из scipy.stats
        post_params --- параметры апостериорных распределений 
                        shape=(размер выборки+1, кол-во параметров)
    '''

    size = len(ml)
    distrs = []
    for n in range(size + 1):
        distrs.append(distr_class(post_params[n]) if np.isscalar(post_params[n])
                      else distr_class(*post_params[n]))

    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(size + 1), [d.mean() for d in distrs], label='Bayes', lw=1.5)
    plt.fill_between(np.arange(size + 1), [d.ppf(0.975) for d in distrs],
                     [d.ppf(0.025) for d in distrs], alpha=0.1)
    plt.plot(np.arange(size) + 1, ml, label='ML', lw=1.5)
    plt.grid(ls=':')
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.legend()
    plt.title(title)
    plt.show()


def bern_posterior_params(sample, a, b):
    ''' Возвращает параметры апостериорного распределения для всех 0 <= n <= len(sample).
        a, b --- параметры априорного распределения.
    '''
    # Beta(a + \sum X_i, b + n - \sum X_i) --- апостериорное распределение для Bern(p)
    a_aposterior = a + sample.cumsum()
    b_aposterior = b + np.arange(1, len(sample) + 1) - sample.cumsum()
    return np.hstack((a, a_aposterior)), np.hstack((b, b_aposterior))


def bern_cumlikelihood(sample):
    """ возвращает массив оценок максимального правдоподобия """
    return sample.cumsum() / np.arange(1, len(sample) + 1)


sample_symmetric = sps.bernoulli(p=0.5).rvs(size=15)
sample_asymmetric = sps.bernoulli(p=0.9).rvs(size=15)
grid = np.linspace(0, 1)
for sample, title in [(sample_symmetric, 'симметричная'), (sample_asymmetric, 'ассиметричная')]:
    for a, b in [(7, 7), (3, 1)]:
        draw_estimations(bern_cumlikelihood(sample), sps.beta, np.column_stack(bern_posterior_params(sample, a, b)), title='{}, априорное: Beta({}, {})'.format(title, a, b), xlim=(0, 15))
