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


plt.figure(figsize=(7, 3.5))

print("load mono")

data1 = np.loadtxt("monovac.xyz", dtype=np.float64)


supercell = 2.855700 * 7

inv = 1 / supercell


def minimage(data):
    data -= supercell * np.floor(data * inv + 0.5)


print("math")


ignore = 3

t1 = data1[ignore::, 0]


disp1 = process(data1[::, 1::])
disp1 -= disp1[0, :]
disp1 = disp1[ignore:, :]
disp1 = disp1 * 1e-10

diffusion(t1, disp1)

disp1 = disp1 * disp1
x1 = np.sum(disp1, axis=1)
diff1 = x1 / (6 * t1)


plotter = plt.loglog

plotter(t1, x1, "-", label=r"Vacancy")


plt.xlabel(r"Time/\si{\second}")
plt.ylabel(r"$\langle x^2 \rangle$/\si{\metre\squared}")


fit = lambda x, a: 6 * a * x


popt, pcov = curve_fit(fit, t1, x1)
print(popt[0], np.sqrt(pcov[0]))

plotter(
    t1,
    6 * 2.25 * 1e-16 * t1,
    "k--",
    label=r"$D = 2.3 \times 10^{-16}$\si{\meter\squared\per\second}",
)


plt.legend()

plt.tight_layout()
plt.savefig(r"/home/cdt1902/dis/thesis/results/Figs/monovac.pdf")

plt.show()
