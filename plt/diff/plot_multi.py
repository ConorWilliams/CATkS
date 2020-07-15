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
    "/home/cdt1902/dis/CATkS/plt/diff/test7_2.xyz", dtype=np.float64
)


ignore = 1

t1 = data1[ignore::, 0]

disp1 = data1[ignore::, 1::] - data1[0, 1::]
disp1 = disp1 * 1e-10

delta = disp1[::, :3] - disp1[::, 3:]
delta *= delta

disp1 *= disp1
x1 = np.sum(disp1[::, :3], axis=1)
x2 = np.sum(disp1[::, 3:], axis=1)
diff1 = x1 / (6 * t1)


plt.plot(t1, x1, label=r"$7^3$ first")
plt.plot(t1, x2, label=r"$7^3$ second")
plt.plot(t1, (x1 + x2) / 2, label=r"$7^3$ mean")


plt.xlabel(r"Time/\si{\nano\second}")
plt.ylabel(r"$\langle x^2 \rangle$/\si{\metre\squared}")

fit = lambda x, a: 6 * a * x

popt, pcov = curve_fit(fit, t1, (x1 + x2) / 2)


print(popt)

plt.legend()

# plt.loglog(t2, 6 * popt[0] * t2, label=r"best fit")


plt.tight_layout()
plt.savefig(r"/home/cdt1902/dis/thesis/validation/Figs/2H_diff.pdf")

plt.show()
