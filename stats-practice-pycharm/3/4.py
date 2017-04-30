import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
from math import *

from sklearn.datasets import load_iris

data = load_iris()
sample = data['data']  # выборка
sample_types = data['target']  # номера компонент смеси

means_of_type = []
covaritaions_of_type = []

for type in [0, 1, 2]:
    # первые 50 ирисок --- первого класса, вторые 50 --- второго, последние 50 --- третьего
    sample_of_type = sample[type * 50:(type + 1) * 50]
    mean_of_type = np.mean(sample_of_type, axis=0)
    covaritaion_of_type = np.cov(sample_of_type.T)
    print(mean_of_type)
    print(covaritaion_of_type)
    print()

    means_of_type.append(mean_of_type)
    covaritaions_of_type.append(covaritaion_of_type)

grid = np.mgrid[4:8:0.01, 1.5:4.5:0.01]
densities = [sps.multivariate_normal.pdf(np.dstack((grid[0], grid[1])), means_of_type[type][:2], covaritaions_of_type[type][:2, :2]) for type in range(3)]
density = np.mean(densities, axis=0)

plt.figure(figsize=(13, 7))
plt.pcolormesh(grid[0], grid[1], density, cmap='Oranges')
plt.scatter(sample[:, 0], sample[:, 1])
CS = plt.contour(grid[0], grid[1], density, [0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
plt.clabel(CS, fontsize=14, inline=1, fmt='%1.2f', cmap='Set3')
plt.show()
