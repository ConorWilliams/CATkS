import matplotlib
import numpy as np
import matplotlib.cm as cm
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


data = np.loadtxt(
    "/home/cdt1902/dis/CATkS/build/sorting2.dat", dtype=np.float64
)

kind = plt.semilogx

f = plt.figure(figsize=(7, 3.5))

kind(
    data[::, 0],
    data[::, 1] / data[::, 0],
    label=r"\texttt{std::sort} random",
    linestyle="-",
    marker="+",
)
kind(
    data[::, 0],
    data[::, 2] / data[::, 0],
    label=r"\texttt{std::sort} near-sorted",
    linestyle="--",
    marker="+",
)
# kind(
#     data[::, 0],
#     data[::, 3] / data[::, 0],
#     label=r"Counting copy random",
#     linestyle="-",
#     marker="+",
# )
# kind(
#     data[::, 0],
#     data[::, 4] / data[::, 0],
#     label=r"Counting copy near-sorted",
#     linestyle="-",
#     marker="+",
# )

kind(
    data[::, 0],
    data[::, 5] / data[::, 0],
    label=r"In-place counting random",
    linestyle="-",
    marker="+",
)

kind(
    data[::, 0],
    data[::, 6] / data[::, 0],
    label=r"In-place counting near-sorted",
    linestyle="--",
    marker="+",
)


plt.legend()
plt.xlabel(r"Number of atoms")
plt.ylabel(r"Time per atom/\si{\nano\second}")
plt.xlim([1e1, 1e5])


plt.tight_layout()
f.savefig(r"/home/cdt1902/dis/thesis/validation/Figs/sort.pdf")


plt.show()
