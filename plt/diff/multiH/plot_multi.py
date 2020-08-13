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

    if len(x) > 100000:
        blk = 5000
    else:
        blk = 750

    div = len(x[0, :]) // 3

    for group in grouper(zip(t, x), blk):
        dx = ((group[-1][1] - group[0][1]) ** 2).sum() / div
        dt = group[-1][0] - group[0][0]
        D.append(dx / (6 * dt))

    tmp = np.asarray(D)

    print(len(x), div)

    print(tmp.mean(), tmp.std(ddof=1) / np.sqrt(len(tmp)))


plt.figure(figsize=(7, 3.5))

data0 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/diff/singleH/1H_7.xyz", dtype=np.float64
)

data1 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/diff/multiH/2H_7.xyz", dtype=np.float64
)

data2 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/diff/multiH/3H_7.xyz", dtype=np.float64
)

data3 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/diff/multiH/4H_7.xyz", dtype=np.float64
)

ignore = 13

plotter = plt.loglog

t0 = data0[ignore::, 0]
disp0 = data0[ignore::, 1::] - data0[0, 1::]
disp0 = disp0 * 1e-10
diffusion(t0, disp0)
disp0 *= disp0
x0 = np.sum(disp0[::, :], axis=1) / (len(disp0[0, :]) // 3)
plotter(t0, x0, label=r"$1$H")

ignore = 3

t1 = data1[ignore::, 0]
disp1 = data1[ignore::, 1::] - data1[0, 1::]
disp1 = disp1 * 1e-10
diffusion(t1, disp1)
disp1 *= disp1
x1 = np.sum(disp1[::, :], axis=1) / (len(disp1[0, :]) // 3)
plotter(t1, x1, label=r"$2$H")


t2 = data2[ignore::, 0]
disp2 = data2[ignore::, 1::] - data2[0, 1::]
disp2 = disp2 * 1e-10
diffusion(t2, disp2)
disp2 *= disp2
x2 = np.sum(disp2[::, :], axis=1) / (len(disp2[0, :]) // 3)
plotter(t2, x2, label=r"$3$H")

t3 = data3[ignore::, 0]
disp3 = data3[ignore::, 1::] - data3[0, 1::]
disp3 = disp3 * 1e-10
diffusion(t3, disp3)
disp3 *= disp3
x3 = np.sum(disp3[::, :], axis=1) / (len(disp3[0, :]) // 3)
plotter(t3, x3, label=r"$4$H")


plt.xlabel(r"Time/\si{\nano\second}")
plt.ylabel(r"$\langle x^2 \rangle$/\si{\metre\squared}")

fit = lambda x, a: 6 * a * x


popt, pcov = curve_fit(fit, t3, x3)


print(popt)

plt.legend()

# plt.loglog(t2, 6 * popt[0] * t2, label=r"best fit")


plt.tight_layout()
plt.savefig(r"/home/cdt1902/dis/thesis/results/Figs/multiH.pdf")

plt.show()
