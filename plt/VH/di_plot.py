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

    blk = 37000

    for group in grouper(zip(t, x), blk):
        dx = ((group[-1][1] - group[0][1]) ** 2).sum()
        dt = group[-1][0] - group[0][0]
        D.append(dx / (6 * dt))

    tmp = np.asarray(D)

    print(len(D), tmp.mean(), tmp.std(ddof=1) / np.sqrt(len(tmp)))

    return tmp.mean()


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

# v2h1 = np.loadtxt("2V1H_super.xyz", dtype=np.float64)
# np.save("V2H1", v2h1)
v2h1 = np.load("V2H1.npy")

# to 185000, post 930000


t = v2h1[:, 0]
e = v2h1[:, 1]


data = process(v2h1[:, 2:])  # unwraps periodic

data -= data[0, :]

data *= 1e-10

v1 = data[:, 0:3]
v2 = data[:, 3:6]
h = data[:, 6:9]

vd = diffusion(t[:185000], v1[:185000])
vd += diffusion(t[:185000], v2[:185000])
hd = diffusion(t[:185000], h[:185000])

vd += diffusion(t[930000:], v1[930000:])
vd += diffusion(t[930000:], v2[930000:])
hd += diffusion(t[930000:], h[930000:])


print(vd / 4, hd / 2)

plt.loglog(t, (h ** 2).sum(axis=1), label="Hydrogen")

plt.loglog(t, (v1 ** 2).sum(axis=1), label="Vacancy 1")

plt.loglog(t, (v2 ** 2).sum(axis=1), label="Vacancy 2")


plt.loglog(
    t,
    6 * 4.29e-16 * t,
    "k--",
    label=r"$D = 4.3 \times 10^{-16}$\si{\meter\squared\per\second}",
)

plt.loglog(
    t,
    6 * 3.7e-18 * t,
    "k-.",
    label=r"$D = 3.7 \times 10^{-18}$\si{\meter\squared\per\second}",
)


plt.xlabel(r"Time/\si{\second}")
plt.ylabel(r"$\langle x^2 \rangle$/\si{\metre\squared}")


plt.legend()

plt.tight_layout()
plt.savefig(r"/home/cdt1902/dis/thesis/results/Figs/VH_di_diff.pdf")

plt.show()
