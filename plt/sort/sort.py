import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{siunitx}")


data = np.loadtxt(
    "/home/cdt1902/dis/CATkS/build/sorting2.dat", dtype=np.float64
)

kind = plt.semilogx

kind(data[::, 0], data[::, 1] / data[::, 0], label="std::sort random")
kind(data[::, 0], data[::, 2] / data[::, 0], label="std::sort near-sorted")
kind(data[::, 0], data[::, 3] / data[::, 0], label="counting::copy")
kind(data[::, 0], data[::, 4] / data[::, 0], label="counting::copy near-sorted")
kind(data[::, 0], data[::, 5] / data[::, 0], label="counting::in-place random")
kind(
    data[::, 0],
    data[::, 6] / data[::, 0],
    label="counting::in-place near-sorted",
)


plt.legend()
plt.xlabel(r"$n$")
plt.ylabel(r"time per item/\si{\nano\second}")
plt.show()
