import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=10 ** 6)

xs = sps.norm.rvs(size=5)
ys = sps.cauchy.rvs(size=1000)
знаменатель = sps.norm(loc=ys[:, np.newaxis]).pdf(xs[np.newaxis, :]).prod(axis=1).mean()

plt.figure(figsize=(15, 7))
grid = np.linspace(-3, 3, 1000)
plt.plot(grid, sps.cauchy.pdf(grid), label='априорная плотность')
plt.plot(grid, sps.cauchy.pdf(grid) * sps.norm(loc=grid[:, np.newaxis]).pdf(xs[np.newaxis, :]).prod(axis=1) / знаменатель, label='апостерионая плотность')
plt.scatter(xs, [-0.1] * 5, label='точки выборки')
plt.scatter(xs.mean(), -0.1, label='среднее точек выборки')
plt.legend()
plt.grid(ls=':')
plt.show()
