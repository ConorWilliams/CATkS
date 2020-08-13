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


supercell = 2.855700 * 7
inv = 1 / supercell


def process(data):
    def sign(dx):
        if dx > supercell * 0.5:
            return -1
        elif dx < -supercell * 0.5:
            return 1
        else:
            return 0

    diff = data[1:, :] - data[:-1, :]

    off = np.vectorize(sign)(diff)

    off = np.cumsum(off, axis=0)

    tmp = data
    tmp[1:] += supercell * off

    return tmp


plt.figure(figsize=(7, 3.5))

v1h1 = np.load("V1H1.npy")
v2h1 = np.load("V2H1.npy")


e1 = v1h1[:980000, 1]
e2 = v2h1[:980000, 1]

print("Mono", len(e1))
print("Di  ", len(e2))


a = plt.hist([e1, e2], 50, label=["Mono-vacancy", "Di-vacancy"], density=False)


plt.ylabel(r"Count")
plt.xlabel(r"Activation energy/\si{\eV}")

plt.legend()

plt.xlim([0, 0.6])
# plt.ylim([0, 81])

plt.yscale("log")

plt.tight_layout()
plt.savefig(r"/home/cdt1902/dis/thesis/results/Figs/VH_hist.pdf")

plt.show()
