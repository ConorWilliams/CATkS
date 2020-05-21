import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from brokenaxes import brokenaxes

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


rawA = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/vac_hist/rawA2.txt", dtype=np.float64
)

rawB = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/vac_hist/rawB2.txt", dtype=np.float64
)

# rawC = np.loadtxt(
#     "/home/cdt1902/dis/CATkS/plt/vac_hist/rawAboost2.txt", dtype=np.float64
# )

bins = 60

plt.figure(figsize=(7, 3.5))

# plt.hist(rawA[::, 1], bins, label="Potential A")
# plt.hist(rawB[::, 1], bins, label="Potential B")
#
# bax = brokenaxes(
#     xlims=((-0.08, 6),), ylims=((0, 150), (800, 1050)), hspace=0.25
# )

b = plt.hist(
    [rawA[::, 1], rawB[::, 1]], bins, label=["Potential A", "Potential B"]
)
plt.xlim([b[-2][0], 6])
plt.yscale("log")
#
plt.legend()
# plt.gca().spines["right"].set_visible(False)
# plt.gca().spines["top"].set_visible(False)
#
plt.xlabel(r"$\Delta E_1$/\si{\eV}")
plt.ylabel(r"Count")


plt.tight_layout()
plt.savefig(r"/home/cdt1902/dis/thesis/validation/Figs/hist.pdf")

plt.show()
