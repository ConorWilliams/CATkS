import numpy as np


supercell = 2.855700 * 7
inv = 1 / supercell


def minimage(data):
    return data - supercell * np.floor(data * inv + 0.5)


v1h1 = np.load("V1H1.npy")


t = v1h1[:, 0]
e = v1h1[:, 1]

v = v1h1[:, 2:5]
h = v1h1[:, 5:8]


delta = np.sqrt((minimage(v - h) ** 2).sum(axis=1)) * 1e-10

bound = 2e-10

print(delta.min(), delta.max())

count, l = np.histogram(delta[:-1], bins=[0, bound, 1])
print("count", count, count[0] / count.sum())

count, l = np.histogram(
    delta[:-1], bins=[0, bound, 1], weights=(t[1:] - t[:-1])
)
print("time", count, count[0] / count.sum())

N_l = 6 * 2 * 7 ** 3 - 24
N_t = 6

# theta = fraction of avaliable sites occupied  =~= time fraction one site occupied over total number
theta_t = (count[0] / count.sum()) / N_t
theta_l = count[1] / count.sum() / N_l

c_l = N_l * theta_l
c_t = N_t * theta_t

oriani = c_l / (c_l + c_t * (1 - theta_t))

print("Oriani Diffusivity", oriani * 7.4e-9)

exp = theta_t / theta_l * (1 - theta_l) / (1 - theta_t)

print(np.log(exp) * 300 * 8.617333262e-5)
