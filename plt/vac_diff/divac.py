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

print("load di")
divac = np.loadtxt("divac.xyz", dtype=np.float64)

supercell = 2.855700 * 7

inv = 1 / supercell


def minimage(data):
    return data - supercell * np.floor(data * inv + 0.5)


print("math")


ignore = 5

# ///////////////////////////////

t2 = divac[ignore::, 0]

x_tmp = divac[::, 1::]
delta = x_tmp[:, :3] - x_tmp[:, 3:]
delta -= supercell * np.floor(0.5 + delta / supercell)
delta *= delta
delta = delta[ignore:, :]
delta = np.sum(delta, axis=1)
delta = np.sqrt(delta)


x2 = process(divac[::, 1::])
x2 -= x2[0, :]
x2 = x2[ignore:, :]
x2 = x2 * 1e-10


xb = []
tb = []
di = True
D = []

life = []

for t, x, d in zip(t2, x2, delta):
    v1 = x[:3]
    v2 = x[3:]

    if d < 5.5:
        xb.append(v1)
        tb.append(t)
    else:
        if len(xb) > 5:
            dx = ((xb[-1] - xb[0]) ** 2).sum()
            dt = tb[-1] - tb[0]
            if dt > 0:
                life.append(dt)
                D.append(dx / (6 * dt))

        xb = []
        tb = []


tmp = np.asarray(D)
print("diff", tmp.mean(), tmp.std(ddof=1) / np.sqrt(len(D)))
tmp = np.asarray(life)
print("life", tmp.mean(), tmp.std(ddof=1) / np.sqrt(len(D)))


x2 *= x2


x2_1 = np.sum(x2[:, :3], axis=1)
x2_2 = np.sum(x2[:, 3:], axis=1)

x2 = (x2_1 + x2_2) * 0.5

plotter = plt.loglog


plotter(t2, x2_1, "-", label=r"Vacancy 1")
plotter(t2, x2_2, "-", label=r"Vacancy 2")


plt.legend()

plt.xlabel(r"Time/\si{\second}")
plt.ylabel(r"$\langle x^2 \rangle$/\si{\metre\squared}")


fit = lambda x, a: 6 * a * x


# popt, pcov = curve_fit(fit, t2, x2)
# print(popt[0])
# plotter(t2, 6 * popt[0] * t2, "k--", label=r"$D = 3.72 \times 10^{-17}$")
#

plt.legend()


plt.tight_layout()
plt.savefig(r"/home/cdt1902/dis/thesis/results/Figs/divac.pdf")

plt.show()
