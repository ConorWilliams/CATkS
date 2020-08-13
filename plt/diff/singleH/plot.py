import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from iteration_utilities import grouper


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

plt.figure(figsize=(7, 3.5))


data1 = np.loadtxt("1H_5.xyz", dtype=np.float64)
data2 = np.loadtxt("1H_7.xyz", dtype=np.float64)
data3 = np.loadtxt("1H_10.xyz", dtype=np.float64)
data4 = np.loadtxt("1H_15.xyz", dtype=np.float64)


def diffusion(t, x):
    # //////
    D = []

    blk = 5000

    for group in grouper(zip(t, x), blk):
        dx = ((group[-1][1] - group[0][1]) ** 2).sum()
        dt = group[-1][0] - group[0][0]
        D.append(dx / (6 * dt))

    tmp = np.asarray(D)

    print(tmp.mean(), tmp.std(ddof=1) / np.sqrt(len(tmp)))


ignore = 20

t1 = data1[ignore::, 0]

disp1 = data1[ignore::, 1::] - data1[0, 1::]
disp1 = disp1 * 1e-10
diffusion(t1, disp1)
disp1 = disp1 * disp1
x1 = np.sum(disp1, axis=1)
diff1 = x1 / (6 * t1)


t2 = data2[ignore::, 0]
disp2 = data2[ignore::, 1::] - data2[0, 1::]
disp2 = disp2 * 1e-10
diffusion(t2, disp2)
disp2 = disp2 * disp2
x2 = np.sum(disp2, axis=1)
diff2 = x2 / (6 * t2)


t3 = data3[ignore::, 0]
disp3 = data3[ignore::, 1::] - data3[0, 1::]
disp3 = disp3 * 1e-10
diffusion(t3, disp3)
disp3 = disp3 * disp3
x3 = np.sum(disp3, axis=1)
diff3 = x3 / (6 * t3)

t4 = data4[ignore::, 0]
disp4 = data4[ignore::, 1::] - data4[0, 1::]
disp4 = disp4 * 1e-10
diffusion(t4, disp4)
disp4 = disp4 * disp4
x4 = np.sum(disp4, axis=1)
diff4 = x4 / (6 * t4)

from scipy.signal import savgol_filter

x1 = savgol_filter(x1, 9, 2)
x2 = savgol_filter(x2, 9, 2)
x3 = savgol_filter(x3, 9, 2)
x4 = savgol_filter(x4, 9, 2)


plotter = plt.loglog

plotter(t1, x1, "-", label=r"$5^3$ unit cells")
plotter(t2, x2, "-", label=r"$7^3$ unit cells")
plotter(t3, x3, "-", label=r"$10^3$ unit cells")
plotter(t4, x4, "-", label=r"$15^3$ unit cells")

plt.legend()

plt.xlabel(r"Time/\si{\second}")
plt.ylabel(r"$\langle x^2 \rangle$/\si{\metre\squared}")

plt.xlim([3e-12, 1e-7])
plt.ylim([1e-20, 1e-14])


fit = lambda x, a: 6 * a * x

plotter(t3, 6 * 7.49 * 1e-9 * t3, "k--", label=r"Experimental")

# popt, pcov = curve_fit(fit, t1, x1)
# print(popt[0], np.sqrt(pcov[0]))
# popt, pcov = curve_fit(fit, t2, x2)
# print(popt[0], np.sqrt(pcov[0]))
# popt, pcov = curve_fit(fit, t3, x3)
# print(popt[0], np.sqrt(pcov[0]))
# popt, pcov = curve_fit(fit, t4, x4)
# print(popt[0], np.sqrt(pcov[0]))

# popt, pcov = curve_fit(fit, t1, x1)
# plotter(t1, 6 * popt[0] * t1, label=f"$5^3$ D {popt[0]}")

# popt, pcov = curve_fit(fit, t2, x2)
# plotter(t2, 6 * popt[0] * t2, label=f"$7^3$ D {popt[0]}")

# popt, pcov = curve_fit(fit, t3, x3)
# plotter(t3, 6 * popt[0] * t3, label=f"$10^3$ D {popt[0]}")
#
# popt, pcov = curve_fit(fit, t4, x4)
# plotter(t4, 6 * popt[0] * t4, label=f"$15^3$ D {popt[0]}")

plt.legend()

# plt.loglog(t2, 6 * popt[0] * t2, label=r"best fit")


plt.tight_layout()
# plt.savefig(r"/home/cdt1902/dis/thesis/validation/Figs/1H_diff.pdf")

plt.show()
