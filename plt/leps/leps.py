import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{siunitx}")


a = 0.05
b = 0.3
c = 0.05
d1 = 4.746
d2 = 4.746
d3 = 3.445
al = 1.942
r0 = 0.742


def Q(r, d):
    return d / 2 * (1.5 * np.exp(-2 * al * (r - r0)) - np.exp(-al * (r - r0)))


def J(r, d):
    return d / 4 * (np.exp(-2 * al * (r - r0)) - 6 * np.exp(-al * (r - r0)))


def E(r1, r2, r3):
    return (
        Q(r1, d1) / (1 + a)
        + Q(r2, d2) / (1 + b)
        + Q(r3, d3) / (1 + c)
        - np.sqrt(
            (J(r1, d1) / (1 + a)) ** 2
            + (J(r2, d2) / (1 + b)) ** 2
            + (J(r3, d3) / (1 + c)) ** 2
            - J(r1, d1) * J(r2, d2) / ((1 + a) * (1 + b))
            - J(r2, d2) * J(r3, d3) / ((1 + b) * (1 + c))
            - J(r1, d1) * J(r3, d3) / ((1 + a) * (1 + c))
        )
    )


rAC = 3.742
kc = 0.2025


def gaus(A, x, y, x0, y0, xi, yi):
    return A * np.exp(-(x - x0) ** 2 / (2 * xi) - (y - y0) ** 2 / (2 * yi))


def V(rAB, rBD):
    return (
        E(rAB, rAC - rAB, rAC) + 2 * kc * (rAB - (rAC / 2 - rBD / 1.154)) ** 2
    )


def foo(x, y):
    return (
        V(x, y)
        + gaus(1.5, x, y, 2.02083, -0.172881, 0.1, 0.35)
        + gaus(6, x, y, 0.8, 2.0, 0.25, 0.7)
    )


print("Python on")

range = 2

xrange = [0.5, 3]
yrange = [-3, 3.3]


delta = 0.025
x = np.arange(*xrange, delta)
y = np.arange(*yrange, delta)

X, Y = np.meshgrid(x, y)

Z = foo(X, Y)

ne = np.loadtxt("ne.dat", dtype=np.float64)
nn = np.loadtxt("nn.dat", dtype=np.float64)
se = np.loadtxt("se.dat", dtype=np.float64)

w = np.loadtxt("w.dat", dtype=np.float64)
s = np.loadtxt("s.dat", dtype=np.float64)

f = plt.figure(figsize=(10, 7))


contours = plt.contour(X, Y, Z, 22, antialiased=True, cmap="gray")

plt.plot(
    nn[::, 0],
    nn[::, 1],
    color="r",
    linestyle="--",
    label="Path 1",
    antialiased=True,
)
plt.quiver(
    nn[::, 0],
    nn[::, 1],
    nn[::, 2],
    nn[::, 3],
    width=0.0035,
    headwidth=1,
    headlength=0,
    pivot="mid",
    label=r"Dimer Axes, $\mathbf{\hat{N}}$",
    antialiased=True,
)

print("1 is", len(nn))

plt.plot(se[::, 0], se[::, 1], color="orange", linestyle="--", label="Dimer 2")
plt.quiver(
    se[::, 0],
    se[::, 1],
    se[::, 2],
    se[::, 3],
    width=0.0035,
    headwidth=1,
    headlength=0,
    pivot="mid",
    antialiased=True,
)

print("2 is", len(se))


plt.plot(ne[::, 0], ne[::, 1], color="b", linestyle="--", label="Dimer 3")
plt.quiver(
    ne[::, 0],
    ne[::, 1],
    ne[::, 2],
    ne[::, 3],
    width=0.0035,
    headwidth=1,
    headlength=0,
    pivot="mid",
    antialiased=True,
)
print("3 is", len(ne))

plt.plot(w[::, 0], w[::, 1], color="green", linestyle="--", label="Dimer 4")
plt.quiver(
    w[::, 0],
    w[::, 1],
    w[::, 2],
    w[::, 3],
    width=0.0035,
    headwidth=1,
    headlength=0,
    pivot="mid",
    antialiased=True,
)

plt.plot(s[::, 0], s[::, 1], color="purple", linestyle="--", label="Dimer 5")
plt.quiver(
    s[::, 0],
    s[::, 1],
    s[::, 2],
    s[::, 3],
    width=0.0035,
    headwidth=1,
    headlength=0,
    pivot="mid",
    antialiased=True,
)


plt.legend()


plt.xlim(xrange)
plt.ylim(yrange)

plt.xlabel(r"$x$/\si{\angstrom}")
plt.ylabel(r"$y$/\si{\angstrom}")

plt.tight_layout()
f.savefig(r"/home/cdt1902/dis/thesis/validation/Figs/2d.pdf")

plt.show()
