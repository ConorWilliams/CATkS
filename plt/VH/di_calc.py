import numpy as np


supercell = 2.855700 * 7
inv = 1 / supercell


def minimage(data):
    return data - supercell * np.floor(data * inv + 0.5)


v2h1 = np.load("V2H1.npy")


t = v2h1[:, 0]
e = v2h1[:, 1]

v1 = v2h1[:, 2:5]
v2 = v2h1[:, 5:8]
h = v2h1[:, 8:11]

sep = np.sqrt((minimage(v1 - v2) ** 2).sum(axis=1))

cin = 0
out = 0

count = np.asarray([0, 0])

for s, v1_, v2_, h_, dt in zip(sep, v1, v2, h, t[1:] - t[:-1]):
    if s < 7:
        c = v1_ + minimage(v2_ - v1_) / 2
        delta = np.sqrt((minimage(c - h_) ** 2).sum())
        if delta < 5:
            cin += dt
            count[0] += 1
        else:
            count[1] += 1
            out += dt

print(cin, out)

print("count", count, count[0] / count.sum())

time = np.asarray([cin, out])

print("time", time, time[0] / time.sum())

N_l = 6 * 2 * 7 ** 3 - (2 * 24 - 4)
N_t = 14

# theta = fraction of avaliable sites occupied  =~= time fraction one site occupied over total number
theta_t = (time[0] / time.sum()) / N_t
theta_l = time[1] / time.sum() / N_l

print("theta_t", theta_t)
print("theta_l", theta_l)

c_l = N_l * theta_l
c_t = N_t * theta_t

oriani = c_l / (c_l + c_t * (1 - theta_t))

print("Oriani Diffusivity", oriani * 7.4e-9)

exp = theta_t / theta_l * (1 - theta_l) / (1 - theta_t)

print(np.log(exp) * 300 * 8.617333262e-5)
