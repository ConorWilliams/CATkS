import matplotlib
import numpy as np
import matplotlib.pyplot as plt

supercell = 2.855700 * 7
inv = 1 / supercell


def minimage(data):
    return data - supercell * np.floor(data * inv + 0.5)


def process(data):
    def sign(dx):
        if dx > supercell * 0.5:
            return -1
        elif dx < -supercell * 0.5:
            return 1
        else:
            return 0

    diff = data[1:, :] - data[:-1, :]

    off = np.vectorize(sign)(diff)

    off = np.cumsum(off, axis=0)

    tmp = data
    tmp[1:] += supercell * off

    return tmp


# 8.2e-18
v2h1 = np.load("V2H1.npy")[:185000, :]


t = v2h1[:, 0]
e = v2h1[:, 1]

v1 = v2h1[:, 2:5]
v2 = v2h1[:, 5:8]
h = v2h1[:, 8:11]


delta = minimage(v1 - v2)
delta = np.sum(delta ** 2, axis=1)
delta = np.sqrt(delta)


x2 = process(v2h1[:, 2:8])
x2 *= 1e-10


xb = []
tb = []
di = True
D = []

life = []

from iteration_utilities import grouper


def diffusion(t, x):
    # //////
    D = []

    blk = 50000

    for group in grouper(zip(t, x), blk):
        dx = ((group[-1][1] - group[0][1]) ** 2).sum()
        dt = group[-1][0] - group[0][0]
        D.append(dx / (6 * dt))

    print(D)

    tmp = np.asarray(D)

    return tmp.mean()


for tg, x, d, c in zip(t, x2, delta, range(len(t))):
    v1 = x[:3]
    v2 = x[3:]

    if d < 6:
        xb.append(v1)
        tb.append(tg)
    else:
        if len(xb) >= 10:
            print(len(xb))
            D.append(diffusion(tb, xb))

        xb = []
        tb = []


tmp = np.asarray(D)
print("diff", tmp.mean(), tmp.std(ddof=1) / np.sqrt(len(D)))
