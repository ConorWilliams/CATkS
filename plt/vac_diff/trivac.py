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


def sign(dx):
    if dx > supercell * 0.5:
        return -1
    elif dx < -supercell * 0.5:
        return 1
    else:
        return 0


def process(data):
    diff = data[1:, :] - data[:-1, :]

    off = np.vectorize(sign)(diff)

    off = np.cumsum(off, axis=0)

    tmp = data
    tmp[1:] += supercell * off

    return tmp


from iteration_utilities import grouper


def diffusion(t, x):
    # //////
    D = []

    blk = 10000

    for group in grouper(zip(t, x), blk):
        dx = ((group[-1][1] - group[0][1]) ** 2).sum()
        dt = group[-1][0] - group[0][0]
        D.append(dx / (6 * dt))

    tmp = np.asarray(D)

    print(tmp.mean(), tmp.std(ddof=1) / np.sqrt(len(tmp)))

    return tmp.mean()


plt.figure(figsize=(7, 3.5))


print("load tri")
trivac = np.loadtxt("trivac.xyz", dtype=np.float64)

supercell = 2.855700 * 7


print("math")

ignore = 3


t3 = trivac[ignore::, 0]

x3 = process(trivac[::, 1::])
x3 -= x3[0, :]
x3 = x3[ignore:, :]
x3 = x3 * 1e-10


x = (x3[:, :3] + x3[:, 3:6] + x3[:, 6:]) / 3

grad = diffusion(t3[1000:], x[1000:])


x3 *= x3

x3_1 = np.sum(x3[:, :3], axis=1)
x3_2 = np.sum(x3[:, 3:6], axis=1)
x3_3 = np.sum(x3[:, 6:], axis=1)

x3 = (x3_1 + x3_2 + x3_3) / 3


plotter = plt.loglog

plotter(t3, x3_1, "-", label=r"Vacancy 1")
plotter(t3, x3_2, "-", label=r"Vacancy 2")
plotter(t3, x3_3, "-", label=r"Vacancy 3")


# plt.legend()

plt.xlabel(r"Time/\si{\second}")
plt.ylabel(r"$\langle x^2 \rangle$/\si{\metre\squared}")


fit = lambda x, a: 6 * a * x


popt, pcov = curve_fit(fit, t3[1000:], x3[1000:])
print(popt[0], np.sqrt(pcov[0]))


popt, pcov = curve_fit(fit, t3, x3)
plotter(
    t3[1000:],
    6 * popt[0] * t3[1000:],
    "k--",
    label=r"$D = 1.8 \times 10^{-13}$",
)


plt.legend()


plt.tight_layout()
plt.savefig(r"/home/cdt1902/dis/thesis/results/Figs/trivac.pdf")

plt.show()
