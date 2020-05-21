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
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.25.txt", dtype=np.float64
)

g5 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.3.txt", dtype=np.float64
)

g6 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.35.txt", dtype=np.float64
)

g7 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.4.txt", dtype=np.float64
)

g8 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.45.txt", dtype=np.float64
)

g9 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.5.txt", dtype=np.float64
)

g10 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/g_init/G_0.55.txt", dtype=np.float64
)


f = plt.figure(figsize=(7, 3.5))

bins = 11

# plt.hist(rawA[::, 1], bins, label="Potential A")
# plt.hist(rawB[::, 1], bins, label="Potential B")

b = plt.hist(
    [
        g1[::, 1],
        g2[::, 1],
        g3[::, 1],
        g4[::, 1],
        g5[::, 1],
        g6[::, 1],
        g7[::, 1],
        g8[::, 1],
        g9[::, 1],
        g10[::, 1],
    ],
    bins,
    label=[
        r"$\sigma = 0.10\si{\angstrom}$",
        r"$\sigma = 0.15\si{\angstrom}$",
        r"$\sigma = 0.20\si{\angstrom}$",
        r"$\sigma = 0.25\si{\angstrom}$",
        r"$\sigma = 0.30\si{\angstrom}$",
        r"$\sigma = 0.35\si{\angstrom}$",
        r"$\sigma = 0.40\si{\angstrom}$",
        r"$\sigma = 0.45\si{\angstrom}$",
        r"$\sigma = 0.50\si{\angstrom}$",
        r"$\sigma = 0.55\si{\angstrom}$",
    ],
)


print(b)

plt.gca().set_xticks(b[-2])

plt.xlim([b[-2][0], b[-2][-1]])

plt.legend()

plt.xlabel(r"$\Delta E_1$/\si{\eV}")
plt.ylabel(r"Count")

plt.tight_layout()
f.savefig(r"/home/cdt1902/dis/thesis/validation/Figs/ghist.pdf")

plt.show()
