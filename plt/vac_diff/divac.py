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


plt.figure(figsize=(7, 3.5))

print("load di")
divac = np.loadtxt("divac.xyz", dtype=np.float64)

supercell = 2.855700 * 7

inv = 1 / supercell


def minimage(data):
    return data - supercell * np.floor((data / supercell) + 0.5)


print("math")


ignore = 3

# ///////////////////////////////

t2 = divac[:, 0]
x2_d = minimage(divac[:, 1:4] - divac[:, 4:])
x2_d *= x2_d
x2_d = np.sqrt(x2_d.sum(axis=1))


plt.semilogx(t2, x2_d)

plt.show()
