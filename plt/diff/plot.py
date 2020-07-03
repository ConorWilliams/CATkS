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


data = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/diff/h_diffusion_5.xyz", dtype=np.float64
)

data = data[::, ::]

disp = data[::, 1::] - data[0, 1::]

disp = disp * 1e-10

disp = disp * disp

sq = np.sum(disp, axis=1)

x = data[::, 0]
y = sq

plt.loglog(x, y, ".")

plt.legend()

plt.xlabel(r"$\Delta E_1$/\si{\eV}")
plt.ylabel(r"Count")


popt, pcov = curve_fit(lambda x, a: a * x, x, y)

print(popt, pcov)


p, V = np.polyfit(x, y, 1, cov=True)

print("x_0: {} +/- {}".format(p[0], np.sqrt(V[0][0])))
print("x_1: {} +/- {}".format(p[1], np.sqrt(V[1][1])))

plt.loglog(x, popt[0] * x)


plt.tight_layout()
# plt.savefig(r"/home/cdt1902/dis/thesis/validation//.pdf")

plt.show()
