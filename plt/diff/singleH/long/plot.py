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

plt.figure(figsize=(7, 3.5))

print("loading")
#
# data1 = np.loadtxt("1H_5_long.xyz", dtype=np.float64)[::1000, ::]
# # data2 = np.loadtxt("1H_7_long.xyz", dtype=np.float64)

import itertools

with open("1H_5_long.xyz") as f_in:
    data1 = np.genfromtxt(
        itertools.islice(f_in, 0, None, 100), dtype=np.float64
    )

print("mathing")

ignore = 20

t1 = data1[ignore::, 0]

disp1 = data1[ignore::, 1::] - data1[0, 1::]
disp1 = disp1 * 1e-10
disp1 = disp1 * disp1
x1 = np.sum(disp1, axis=1)
diff1 = x1 / (6 * t1)

#
# t2 = data2[ignore::, 0]
# disp2 = data2[ignore::, 1::] - data2[0, 1::]
# disp2 = disp2 * 1e-10
# disp2 = disp2 * disp2
# x2 = np.sum(disp2, axis=1)
# diff2 = x2 / (6 * t2)

print("plotting")


plotter = plt.loglog

plotter(t1, x1, "-", label=r"$5^3$ unit cells")
# plotter(t2, x2, "-", label=r"$7^3$ unit cells")


plt.legend()

plt.xlabel(r"Time/\si{\second}")
plt.ylabel(r"$\langle x^2 \rangle$/\si{\metre\squared}")


fit = lambda x, a: 6 * a * x

plotter(t1, 6 * 7.49 * 1e-9 * t1, "k--", label=r"Experimental")

popt, pcov = curve_fit(fit, t1, x1)
print(popt[0], np.sqrt(pcov[0]))
# popt, pcov = curve_fit(fit, t2, x2)
# print(popt[0], np.sqrt(pcov[0]))


# popt, pcov = curve_fit(fit, t1, x1)
# plotter(t1, 6 * popt[0] * t1, label=f"$5^3$ D {popt[0]}")

# popt, pcov = curve_fit(fit, t2, x2)
# plotter(t2, 6 * popt[0] * t2, label=f"$7^3$ D {popt[0]}")


plt.legend()

# plt.loglog(t2, 6 * popt[0] * t2, label=r"best fit")


plt.tight_layout()
plt.savefig(r"/home/cdt1902/dis/thesis/validation/Figs/1H_diff_long.pdf")

plt.show()
