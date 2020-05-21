import matplotlib
import numpy as np
import matplotlib.pyplot as plt

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


g1 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.1.txt", dtype=np.float64
)

g2 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.15.txt", dtype=np.float64
)

g3 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.2.txt", dtype=np.float64
)

g4 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.225.txt", dtype=np.float64
)

g5 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.25.txt", dtype=np.float64
)

g6 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.275.txt", dtype=np.float64
)

g7 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.3.txt", dtype=np.float64
)

g8 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.325.txt", dtype=np.float64
)

g9 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.35.txt", dtype=np.float64
)

g10 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.375.txt", dtype=np.float64
)

g11 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.4.txt", dtype=np.float64
)

g12 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.45.txt", dtype=np.float64
)

g13 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.5.txt", dtype=np.float64
)

g14 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.55.txt", dtype=np.float64
)


f = plt.figure(figsize=(7, 3.5))


# plt.hist(rawA[::, 1], bins, label="Potential A")
# plt.hist(rawB[::, 1], bins, label="Potential B")


succes = [
    len(g[::, 1]) / 5
    for g in [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14]
]

x = [
    0.1,
    0.15,
    0.2,
    0.225,
    0.25,
    0.275,
    0.3,
    0.325,
    0.35,
    0.375,
    0.4,
    0.45,
    0.5,
    0.55,
]


plt.plot(x, succes, "+k")


plt.xlabel(r"$\sigma$/\si{\angstrom}")
plt.ylabel(r"Percentage successful SP searches")

plt.ylim([0, 60])

plt.tight_layout()
f.savefig(r"/home/cdt1902/dis/thesis/validation/Figs/g_succ.pdf")

plt.show()
