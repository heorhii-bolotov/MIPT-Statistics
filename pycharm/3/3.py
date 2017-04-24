import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
from math import *

# чтобы сумма была больше ста, нужна взять чуть больше 25 штук
# для простоты возьмём 100
ξs = sps.expon(scale=4).rvs(size=100)
assert ξs.sum() > 100

t = 100
plt.figure(figsize=(15, 5))
for λ, color in [(1 / 10, 'red'), (1 / 4, 'green'), (1 / 2, 'cyan'), (1, 'magenta')]:
    for n_s in range(len(ξs)):
        s = ξs[:n_s].sum()
        if s + ξs[n_s] > t:
            break
        x1 = s
        x2 = s + ξs[n_s]
        y1 = λ * (t - x1) + n_s
        y2 = λ * (t - x2) + n_s
        plt.plot([x1, x2], [y1, y2], color=color, label='λ = {}'.format(λ) if n_s == 0 else '')
n_100 = min([n for n in range(len(ξs)) if ξs[:n].sum() >= t])
plt.hlines(n_100, 0, t, label='$N_{100}$')
plt.legend()
plt.grid(ls=':')
plt.show()
