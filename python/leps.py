import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

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


range = 2

xrange = [0.45, 3.3]
yrange = [-3, 3.7]

delta = 0.025
x = np.arange(*xrange, delta)
y = np.arange(*yrange, delta)

X, Y = np.meshgrid(x, y)

Z = foo(X, Y)

x = np.asarray([0, 1])
y = np.asarray([0, 1])

# levels = np.arange(-20, 20, 1)

# plt.contourf(X, Y, Z, 30, cmap="cividis")
plt.contour(X, Y, Z, 30, cmap="gnuplot2")
plt.plot(x, y, "k+")

plt.xlim(xrange)
plt.ylim(yrange)

plt.title("$2$D test function")

plt.show()
