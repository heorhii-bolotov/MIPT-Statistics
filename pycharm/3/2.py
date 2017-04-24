import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
from math import *


def checkInverse():
    a = np.array([[10, 8], [8, 10]])
    b = np.array([[5 / 18, -2 / 9], [-2 / 9, 5 / 18]])
    print(a)
    print(b)
    print(a @ b)


grid = np.linspace(-10, 10, 1000)
plt.figure(figsize=(15, 5))
for y in [-3, 0, 1, 5]:
    f_ξ1_ξ2 = sqrt(5) / 6 * np.exp(-5 / 36 * grid ** 2 - 4 / 45 * y ** 2 + 2 / 9 * grid * y)
    plt.plot(grid, f_ξ1_ξ2, linewidth=3, label='y = {}'.format(y))
    plt.legend(fontsize=16)
    plt.grid(ls=':')
plt.show()
