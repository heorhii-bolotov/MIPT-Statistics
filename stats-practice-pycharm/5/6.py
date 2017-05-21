import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import tabulate


def print_table(values, минимальное_значение_p_при_котором_не_отвергаем=0.5):
    table = [['n', 'p', '$\sum X_i$', r'$c_\alpha$', 'p-value', 'отвергаем?', 'должны отвергать?']]
    for n, p in values:
        sample = sps.bernoulli(p).rvs(n)
        t = sample.sum()
        c_alpha = sps.binom(n, 0.5).ppf(1 - 0.05)
        p_value = sps.binom(n, 0.5).sf(t)
        отвергаем = 'да' if t > c_alpha else 'нет'
        должны_отвергать = 'нет' if p <= минимальное_значение_p_при_котором_не_отвергаем else 'да'
        line = [n, p, t, c_alpha, '{:.3f}'.format(p_value), отвергаем, должны_отвергать]
        table.append(line)
    display(HTML(tabulate.tabulate(table, tablefmt='html')))


# проверка работоспособности критерия
ns = [10, 20, 50, 100]
ps = np.linspace(0.5, 1, 6)
values = [(n, p) for n in ns for p in ps]
print_table(values)

#
values = [(5, 0.75)] * 10 + [(10 ** 5, 0.51)] * 10
print_table(values)

#
p0 = 0.6
plt.figure(figsize=(13, 10))
plt.xlabel('p')
plt.ylabel('мощность критерия')
for n in [10, 20, 50, 100, 150, 200, 1000]:
    c_alpha = sps.binom(n, 0.5).ppf(1 - 0.05)
    grid = np.linspace(0.5, 1)
    plt.plot(grid, sps.binom(n, grid).sf(c_alpha), label='n={}'.format(n))
plt.vlines(p0, 0, 1, label='линия $p^*$')
plt.hlines(0.8, 0.5, 1, color='cyan', label='линия мощности 0.8')
plt.legend()
plt.show()
