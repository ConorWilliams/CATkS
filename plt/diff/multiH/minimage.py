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
        blk = 500

    div = len(x[0, :]) // 3

    for group in grouper(zip(t, x), blk):
        dx = ((group[-1][1] - group[0][1]) ** 2).sum() / div
        dt = group[-1][0] - group[0][0]
        D.append(dx / (6 * dt))

    tmp = np.asarray(D)

    print(len(x), div, t[-1])

    print(tmp.mean(), tmp.std(ddof=1) / np.sqrt(len(tmp)))


supercell = 2.855700 * 7
inv = 1 / supercell


def minimage(data):
    return data - supercell * np.floor(data * inv + 0.5)


plt.figure(figsize=(7, 3.5))


# data1 = np.loadtxt(
#     "/home/cdt1902/dis/CATkS/plt/diff/multiH/2H_7.xyz", dtype=np.float64
# )
#
# np.save("data1", data1)

data1 = np.load("data1.npy")


# data2 = np.loadtxt(
#     "/home/cdt1902/dis/CATkS/plt/diff/multiH/3H_7.xyz", dtype=np.float64
# )
#
# np.save("data2", data2)
#
data2 = np.load("data2.npy")


# data3 = np.loadtxt(
#     "/home/cdt1902/dis/CATkS/plt/diff/multiH/4H_7.xyz", dtype=np.float64
# )
#
# np.save("data3", data3)

data3 = np.load("data3.npy")


plotter = plt.loglog

bins = 150


list = []

from itertools import product

offsets = [
    [0.5, 0.25, 0],
    [0.25, 0.5, 0],
    [0.5, 0, 0.25],
    [0.25, 0, 0.5],
    [0, 0.5, 0.25],
    [0, 0.25, 0.5],
]
#
offsets = [np.asarray(x) for x in offsets]


for r in product([0, 1, 2, 3, 4, 5, 6, 7], repeat=3):
    for o in offsets:
        list.append(np.asarray(r) + np.asarray([0.5 + 0.5 + 0.5]) + o)
        list.append(np.asarray(r) + o)


dist = np.asarray(list)

dist -= -np.asarray([0.25, 0.5, 0])

dist *= 2.8557

dist = minimage(dist)

dist = np.sqrt((dist * dist).sum(axis=1))


rdf, l = np.histogram(dist, density=False, bins=bins, range=(0, 17.31))


rdf = rdf[1:]

r = (l[2:] + l[1:-1]) * 0.5

rdf = (
    rdf
    * supercell ** 3
    / (4 * np.pi * r ** 2 * (len(dist) - 1) * (r[1] - r[0]))
)


plt.plot(r, rdf, "k--", label="Lattice", dashes=(1, 1))


def histbin(time, positions, label=None):
    from itertools import combinations

    out = np.array([])

    dt = np.array([])

    for a, b in combinations(positions, r=2):
        out = np.hstack(
            [out, np.sqrt((minimage(a[:-1] - b[:-1]) ** 2).sum(axis=1))]
        )
        dt = np.hstack([dt, time[1:] - time[:-1]])

    print(out.shape)
    print(dt.shape)

    rdf, l = np.histogram(
        out, density=False, bins=bins, range=(0, 17.31), weights=dt
    )

    r = (l[1:] + l[:-1]) * 0.5

    N = len(positions)
    #
    # print(time[-1])

    rdf = (
        rdf
        * supercell ** 3
        / (4 * np.pi * time[-1] * N * (N - 1) * r ** 2 * (l[1] - l[0]))
    )

    plt.plot(r, rdf, label=label)

    return out


histbin(data1[:, 0], [data1[:, 1:4], data1[:, 4:7]], label="2H")
histbin(data2[:, 0], [data2[:, 1:4], data2[:, 4:7], data2[:, 7:10]], label="3H")
histbin(
    data3[:, 0],
    [data3[:, 1:4], data3[:, 4:7], data3[:, 7:10], data3[:, 10:13]],
    label="4H",
)


plt.legend()

# ////////////////////////////////

x1, x2 = 0, 13


plt.xlim([x1, x2])
plt.xticks(np.arange(x1, x2 + 1, 1.0))

plt.xlabel(r"r/\si{\angstrom}")
plt.ylabel(r"$g(r)$")


# plt.loglog(t2, 6 * popt[0] * t2, label=r"best fit")


plt.tight_layout()
plt.savefig(r"/home/cdt1902/dis/thesis/results/Figs/histH.pdf")

plt.show()
