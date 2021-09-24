import numpy as np
import os
from green import green_function as gf


# Will be used for calculation of alpha_mat
# Calculates the influence of panel Qind in panQ at panP_cen points


def int_G_dGbydn(panP, panQ, Qind, frq):
    ulen = 1.0

    gxp = panP.Centroids_Global[:, 0]
    gyp = panP.Centroids_Global[:, 1]
    gzp = panP.Centroids_Global[:, 2]

    gxq = panQ.Centroids_Global[Qind, 0]
    gyq = panQ.Centroids_Global[Qind, 1]
    gzq = panQ.Centroids_Global[Qind, 2]

    x = gxp / ulen
    y = gyp / ulen
    z = gzp / ulen

    xs = gxq / ulen
    ys = gyq / ulen
    zs = gzq / ulen

    dS = panQ.Area[Qind]

    nx = panP.UnitVecs_Global[:, 2, 0]
    ny = panP.UnitVecs_Global[:, 2, 1]
    nz = panP.UnitVecs_Global[:, 2, 2]

    G, Gx, Gy, Gz = gf.green(x, y, z, xs, ys, zs, frq, ulen)
    dGdn = Gx * nx + Gy * ny + Gz * nz

    int_G = G * dS
    int_dGdn = dGdn * dS

    return int_G, int_dGdn


# Will be used for calculation of beta_mat
# Calculates the influence of panel Qind in panQ at panP_cen points
# def int_G(panP_cen, panQ, Qind, frq):
#     return np.zeros(panP_cen.shape[0])


def debug_green():
    x = np.array([0.5, 2.5, 5.0, 7.5, 10.0])
    y = x - 3.0
    z = -x

    xs = 0.0
    ys = 0.0
    zs = 0.0

    frq = 0.5
    ulen = 1.0

    G, Gx, Gy, Gz = gf.green(x, y, z, xs, ys, zs, frq, ulen)

    print('Green function from Hydra Python')
    print(G)
    print('\n')

    print('dGdx from Hydra Python')
    print(Gx)
    print('\n')

    print('dGdy from Hydra Python')
    print(Gy)
    print('\n')

    print('dGdz from Hydra Python')
    print(Gz)
    print('\n')


if __name__ == "__main__":
    print("Visharad")
    debug_green()
