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


data1 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/diff/1H_5.xyz", dtype=np.float64
)


data2 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/diff/1H_7.xyz", dtype=np.float64
)

ignore = 25

data1 = data1[ignore::, ::]
t1 = data1[::, 0]

disp1 = data1[::, 1::] - data1[0, 1::]
disp1 = disp1 * 1e-10
disp1 = disp1 * disp1
x1 = np.sum(disp1, axis=1)
diff1 = x1 / (6 * t1)


data2 = data2[ignore::, ::]
t2 = data2[::, 0]
disp2 = data2[::, 1::] - data2[0, 1::]
disp2 = disp2 * 1e-10
disp2 = disp2 * disp2
x2 = np.sum(disp2, axis=1)
diff2 = x2 / (6 * t2)


plt.plot(t1, diff1, label=r"$5^3$ unit cells")
plt.plot(t2, diff2, label=r"$7^3$ unit cells")


plt.legend()

plt.xlabel(r"Time/\si{\nano\second}")
plt.ylabel(r"$\langle x^2 \rangle$/\si{\metre\squared}")

fit = lambda x, a: 6 * a * x

popt, pcov = curve_fit(fit, t2, x2)

print(popt)
print(pcov)

print(popt[0], np.sqrt(pcov)[0, 0])

# plt.loglog(t2, 6 * popt[0] * t2, label=r"best fit")


plt.tight_layout()
plt.savefig(r"/home/cdt1902/dis/thesis/validation/Figs/1H_diff.pdf")

plt.show()
