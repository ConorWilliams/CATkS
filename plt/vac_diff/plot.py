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
    tmp[1:, :] += supercell * off

    return tmp


plt.figure(figsize=(7, 3.5))

print("load di")
# divac = np.loadtxt("divac.xyz", dtype=np.float64)
# np.save("divac", divac)
divac = np.load("divac.npy")

print("load tri")
# trivac = np.loadtxt("trivac.xyz", dtype=np.float64)
# np.save("trivac", trivac)
trivac = np.load("trivac.npy")


def minimage(data):
    return data - supercell * np.floor(data * inv + 0.5)


print("math")


supercell = 2.855700 * 7

# ///////////////////////////////


delta = divac[:, 1:4] - divac[:, 4:7]
delta -= supercell * np.floor(0.5 + delta / supercell)
delta = delta * delta

delta = np.sqrt(delta.sum(axis=1))

delta = delta[:-1]

time = divac[1:, 0] - divac[:-1, 0]

time /= time.sum()

# ///////////////////

delta12 = trivac[:, 1:4] - trivac[:, 4:7]
delta12 -= supercell * np.floor(0.5 + delta12 / supercell)
delta12 = delta12 * delta12
delta12 = np.sqrt(delta12.sum(axis=1))
delta12 = delta12[:-1]

time12 = trivac[1:, 0] - trivac[:-1, 0]

time12 /= time12.sum()

delta23 = trivac[:, 4:7] - trivac[:, 7:10]
delta23 -= supercell * np.floor(0.5 + delta23 / supercell)
delta23 = delta23 * delta23
delta23 = np.sqrt(delta23.sum(axis=1))
delta23 = delta23[:-1]

time23 = trivac[1:, 0] - trivac[:-1, 0]
time23 /= time23.sum()

delta13 = trivac[:, 1:4] - trivac[:, 7:10]
delta13 -= supercell * np.floor(0.5 + delta13 / supercell)
delta13 = delta13 * delta13
delta13 = np.sqrt(delta13.sum(axis=1))
delta13 = delta13[:-1]

time13 = trivac[1:, 0] - trivac[:-1, 0]
time13 /= time13.sum()

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

sdelta = np.concatenate((delta12, delta23, delta13))
stime = np.concatenate((time12, time23, time13)) / 3

c = ax2.hist(
    [delta, sdelta],
    50,
    weights=[time, stime],
    density=False,
    label=[r"Di-vacancy", r"Tri-vacancy"],
)
ax2.legend()
ax2.set_ylabel(r"Time fraction")
ax2.set_xticks(c[1])
print(c[1])

plt.xticks(rotation=90)

ax1.hist(
    [delta, sdelta], 50, density=True, label=[r"Di-vacancy", r"Tri-vacancy"]
)

ax1.legend()
ax1.set_ylabel(r"Normalised count")

plt.xlim([2, 10])
plt.xlabel(r"Vacancy-Vacancy separation/\si{\angstrom}")


plt.tight_layout()
plt.savefig(r"/home/cdt1902/dis/thesis/results/Figs/vac_sep.pdf")

plt.show()
