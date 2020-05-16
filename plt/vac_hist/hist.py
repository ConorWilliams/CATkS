import matplotlib
import numpy as np
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{siunitx}")


rawA = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/vac_hist/rawA2.txt", dtype=np.float64
)

rawB = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/vac_hist/rawB2.txt", dtype=np.float64
)


plt.hist(rawA[::, 1], 50)
plt.hist(rawB[::, 1], 50)

plt.xlim(0, 6)

plt.xlabel(r"$\Delta E_1$/\si{\eV}")
plt.ylabel(r"Count")

plt.tight_layout()
plt.savefig(r"/home/cdt1902/dis/thesis/validation/Figs/hist.pdf")

plt.show()
