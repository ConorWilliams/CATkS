import matplotlib
import numpy as np
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{siunitx}")

raw = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/vac_hist/raw.txt", dtype=np.float64
)

raw1 = np.loadtxt(
    "/home/cdt1902/dis/CATkS/plt/vac_hist/raw1.txt", dtype=np.float64
)

energies = []

for row in raw:
    energies.append(row[1])

for row in raw1:
    energies.append(row[1])

print(len(energies))

plt.hist(energies, 90, histtype="bar")

plt.xlim(0, 6)

plt.xlabel(r"$\Delta E_1$/\si{\eV}")
plt.ylabel(r"Count")

plt.tight_layout()
plt.savefig(r"/home/cdt1902/dis/thesis/validation/Figs/hist.pdf")

plt.show()
