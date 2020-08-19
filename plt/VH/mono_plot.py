import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc(
    "text.latex",
    preamble=r"""\RequirePackage[utf8]{inputenc}
                 \RequirePackage[T1]{fontenc}
                 \RequirePackage[final]{microtype}
                 \RequirePackage{amsfonts}
                 \RequirePackage{amsmath}
                 \RequirePackage{amssymb}
                 \usepackage{siunitx}
                 \PassOptionsToPackage{full}{textcomp}
                 \usepackage{textcomp}
                 \usepackage{newtxtext}
                 \usepackage[subscriptcorrection,vvarbb]{newtxmath}
                 \usepackage{bm}""",
)

from iteration_utilities import grouper


def diffusion(t, x):
    # //////
    D = []

    blk = 200000

    div = len(x[0, :]) // 3

    for group in grouper(zip(t, x), blk):
        dx = ((group[-1][1] - group[0][1]) ** 2).sum() / div
        dt = group[-1][0] - group[0][0]
        D.append(dx / (6 * dt))

    tmp = np.asarray(D)

    print(len(D), div, tmp.mean(), tmp.std(ddof=1) / np.sqrt(len(tmp)))


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


plt.figure(figsize=(7, 3.5))

v1h1 = np.load("V1H1.npy")


t = v1h1[:, 0]
e = v1h1[:, 1]


data = process(v1h1[:, 2:]) * 1e-10  # unwraps periodic

data -= data[0, :]

v = data[:, :3]
h = data[:, 3:]

diffusion(t, v)
diffusion(t, h)

plt.plot(t, (h ** 2).sum(axis=1), label="Hydrogen")

plt.loglog(
    t,
    6 * 2.118e-15 * t,
    "k-.",
    label=r"$D = 2.1 \times 10^{-15}$\si{\meter\squared\per\second}",
)


plt.plot(t, (v ** 2).sum(axis=1), label="Mono-vacancy")


plt.loglog(
    t,
    6 * 1.187e-17 * t,
    "k--",
    label=r"$D = 1.2 \times 10^{-17}$\si{\meter\squared\per\second}",
)


plt.xlabel(r"Time/\si{\second}")
plt.ylabel(r"$\langle x^2 \rangle$/\si{\metre\squared}")


plt.legend()

plt.tight_layout()
plt.savefig(r"/home/cdt1902/dis/thesis/results/Figs/VH_diff.pdf")

plt.show()
