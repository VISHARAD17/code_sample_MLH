import numpy as np
import module_shared as sh
import module_radscat_shared as rs
from scipy.linalg.lapack import zgetrf, zgetrs
from module_one_by_r import int_dbydn_one_by_r, int_one_by_r
from module_green import int_G_dGbydn
import csv
import matplotlib.pyplot as plt
import os


def calculate_radiation_scattering():
    initialize()
    calculate_frq_indep_alpha_beta()

    for i in range(rs.nfrq):
        print('--------------------------------------------')
        print('Frequency', i + 1, ' of ', rs.nfrq, ': ', rs.frq[i], ' rad/s')
        print('--------------------------------------------')

        calculate_frq_dep_alpha_beta(rs.frq[i])
        calculate_vn(rs.frq[i])

        calculate_sigma()
        calculate_phi()

        calculate_radiation(i)
        calculate_scattering(i)
    exit()


def test():
    dirctry = r'C:/Users/Harikishore P/Desktop/Running works/ASRY project/simdyn work dir/results/'

    # Writing the data to csv sheet
    with open(os.path.join(dirctry, 'sheets/sctfrc.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['FRQ', 'BETA', 'Mode', 'Mode(F)',
                         'Pha(F)', 'Re(F)', 'Im(F)'])
        for i in range(rs.nfrq):
            for j in range(rs.nbet):
                for k in range(6 * rs.nVessels):
                    writer.writerow([rs.frq[i] / (2 * np.pi), rs.bet[j] * 180 / np.pi, k + 1, abs(rs.ScatFrc[i, k, j]),
                                     np.arctan2(rs.ScatFrc[i, k, j].imag, rs.ScatFrc[i, k, j].real),
                                     rs.ScatFrc[i, k, j].real, rs.ScatFrc[i, k, j].imag])
    csvfile.close()

    # loading ground truth values
    actual_912 = np.zeros((rs.nfrq, 6 * rs.nVessels, rs.nbet), dtype=complex)
    f = open(os.path.join(
        dirctry, 'box_barge_912_ForceFroudeKrylov.csv'), newline='')
    reader = csv.reader(f, delimiter=',')
    temp = next(reader)
    temp = next(reader)  # Skipping first two rows in the sheet

    for i in range(rs.nfrq):
        for j in range(rs.nbet):
            for k in range(6 * rs.nVessels):
                actual_912[i, k, j] = float(next(reader)[3])
    f.close()

    actual_1235 = np.zeros((rs.nfrq, 6 * rs.nVessels, rs.nbet), dtype=complex)
    f = open(os.path.join(
        dirctry, 'box_barge_1235_ForceFroudeKrylov.csv'), newline='')
    reader = csv.reader(f, delimiter=',')
    temp = next(reader)
    temp = next(reader)  # Skipping first two rows in the sheet

    for i in range(rs.nfrq):
        for j in range(rs.nbet):
            for k in range(6 * rs.nVessels):
                actual_1235[i, k, j] = float(next(reader)[3])
    f.close()

    # Non dimensionlizing scatFrc and rs.frq
    # rs.ScatFrc /= sh.rho * sh.g
    # rs.frq *= np.sqrt(1 / sh.g)

    # Generating Froude Krylov plots for every wave dir and frequency

    # Plotting the forces
    for j in range(rs.nbet):
        for k in range(6 * rs.nVessels):
            print("force " + str(k + 1) + " bet " + str(j + 1))
            try:
                plt.plot(rs.frq, abs(rs.ScatFrc[:, k, j]), label="computed",
                         linestyle="-", marker='*', color='b')
            except Exception as e:
                print(e)
                pass
            try:
                plt.plot(rs.frq, abs(actual_1235[:, k, j]), color='r',
                         label="actual", linestyle="-")
            except Exception as e:
                print(e)
                pass
            plt.legend()
            plt.xlabel('w*sqrt(L/g)')
            plt.ylabel('|F' + str(j + 1) + '|/rho*g*A*L**2')
            plt.title("F" + str(k + 1) + " bet = " +
                      str(rs.bet[j] * 180 / np.pi))
            plt.savefig(
                os.path.join(dirctry, "plots/F" + str(k + 1) + " bet = " + str(rs.bet[j] * 180 / np.pi) + '.png'))
            plt.clf()


def initialize():
    rs.nVessels = sh.sim.nVessels
    rs.PanelRange = np.cumsum(
        np.array([vessel.Grid.nUWPan for vessel in sh.sim.Vessels]))
    rs.nPan = rs.PanelRange[rs.nVessels - 1]

    # Initialize matrices
    rs.alpha_mat = np.zeros((rs.nPan, rs.nPan), dtype=complex)
    rs.beta_mat = np.zeros((rs.nPan, rs.nPan), dtype=complex)
    rs.alpha_mat_com = np.zeros((rs.nPan, rs.nPan), dtype=complex)
    rs.beta_mat_com = np.zeros((rs.nPan, rs.nPan), dtype=complex)
    rs.vn = np.zeros((rs.nPan, 6 * rs.nVessels + rs.nbet), dtype=complex)
    rs.sigma = np.zeros((rs.nPan, 6 * rs.nVessels + rs.nbet), dtype=complex)
    rs.phi = np.zeros((rs.nPan, 6 * rs.nVessels + rs.nbet), dtype=complex)
    rs.phiI = np.zeros((rs.nPan, rs.nbet), dtype=complex)
    rs.n_terms = np.zeros((rs.nPan, 6 * rs.nVessels))
    rs.cen = np.zeros((rs.nPan, 3))
    rs.globnorm = np.zeros((rs.nPan, 3))
    rs.pan_area = np.zeros(rs.nPan)
    rs.AddMass = np.zeros((rs.nfrq, 6 * rs.nVessels, 6 * rs.nVessels))
    rs.RadDamp = np.zeros((rs.nfrq, 6 * rs.nVessels, 6 * rs.nVessels))
    rs.ScatFrc = np.zeros((rs.nfrq, 6 * rs.nVessels, rs.nbet), dtype=complex)
    rs.ScatFrc_Haskind = np.zeros(
        (rs.nfrq, 6 * rs.nVessels, rs.nbet), dtype=complex)
    rs.FKFrc = np.zeros((rs.nfrq, 6 * rs.nVessels, rs.nbet), dtype=complex)


def calculate_frq_indep_alpha_beta():
    for body1 in range(rs.nVessels):  # Influencing body
        xbody = sh.sim.Vessels[body1].Grid.xbody

        # Extract local normals and centroids of influencing body
        nx = sh.sim.Vessels[body1].Grid.UWPanels.UnitVecs[:, 2, 0]
        ny = sh.sim.Vessels[body1].Grid.UWPanels.UnitVecs[:, 2, 1]
        nz = sh.sim.Vessels[body1].Grid.UWPanels.UnitVecs[:, 2, 2]

        xc = sh.sim.Vessels[body1].Grid.UWPanels.Centroids[:, 0]
        yc = sh.sim.Vessels[body1].Grid.UWPanels.Centroids[:, 1]
        zc = sh.sim.Vessels[body1].Grid.UWPanels.Centroids[:, 2]

        if body1 == 0:
            indx1a = 0
        else:
            indx1a = rs.PanelRange[body1 - 1]
        indx1b = rs.PanelRange[body1]

        # Store local normals of all bodies in n_terms matrix in module_radscat_shared.py
        rs.n_terms[indx1a:indx1b, body1 * 6 + 0] = nx
        rs.n_terms[indx1a:indx1b, body1 * 6 + 1] = ny
        rs.n_terms[indx1a:indx1b, body1 * 6 + 2] = nz
        rs.n_terms[indx1a:indx1b, body1 * 6 + 3] = yc * nz - zc * ny
        rs.n_terms[indx1a:indx1b, body1 * 6 + 4] = zc * nx - xc * nz
        rs.n_terms[indx1a:indx1b, body1 * 6 + 5] = xc * ny - yc * nx

        # storing global coordinates of centroids of panels in cen
        rs.cen[indx1a:indx1b,
        0] = sh.sim.Vessels[body1].Grid.UWPanels.Centroids_Global[:, 0]
        rs.cen[indx1a:indx1b,
        1] = sh.sim.Vessels[body1].Grid.UWPanels.Centroids_Global[:, 1]
        rs.cen[indx1a:indx1b,
        2] = sh.sim.Vessels[body1].Grid.UWPanels.Centroids_Global[:, 2]

        # storing global coordinates of normals of panels in globnorm
        rs.globnorm[indx1a:indx1b,
        0] = sh.sim.Vessels[body1].Grid.UWPanels.UnitVecs_Global[:, 2, 0]
        rs.globnorm[indx1a:indx1b,
        1] = sh.sim.Vessels[body1].Grid.UWPanels.UnitVecs_Global[:, 2, 1]
        rs.globnorm[indx1a:indx1b,
        2] = sh.sim.Vessels[body1].Grid.UWPanels.UnitVecs_Global[:, 2, 2]

        # Store panel areas of all bodies in pan_area in module_radscat_shared.py
        rs.pan_area[indx1a:indx1b] = sh.sim.Vessels[body1].Grid.UWPanels.Area

        for body2 in range(rs.nVessels):  # Influenced body

            print('Calculating frequency independent influence of body ',
                  body1 + 1, 'on ', body2 + 1)

            # Influencing panel Q(xi,eta,zeta)
            for m in range(sh.sim.Vessels[body1].Grid.nUWPan):

                if body1 == 0:
                    indx1 = m
                else:
                    indx1 = m + rs.PanelRange[body1 - 1]

                if body2 == 0:
                    indx2a = 0
                else:
                    indx2a = rs.PanelRange[body2 - 1]
                indx2b = rs.PanelRange[body2]

                # Corrected rs.panP to just P
                panP = sh.sim.Vessels[body2].Grid.UWPanels
                panQ = sh.sim.Vessels[body1].Grid.UWPanels
                panQ_Image = sh.sim.Vessels[body1].Grid.UWPanels_Image

                # 4 arguments to 3 for module_one_by_r functions
                rs.alpha_mat_com[indx2a:indx2b, indx1] = (1 / (2.0 * np.pi)) * (
                        int_dbydn_one_by_r(panP, panQ, m)
                        + int_dbydn_one_by_r(panP, panQ_Image, m))

                rs.beta_mat_com[indx2a:indx2b, indx1] = (1 / (4.0 * np.pi)) * (
                        int_one_by_r(panP, panQ, m)
                        + int_one_by_r(panP, panQ_Image, m))

            for i in range(rs.nPan):
                rs.alpha_mat_com[i, i] = - 1.0

            if np.any(np.isnan(rs.alpha_mat_com[indx2a:indx2b, indx1])):
                print("nan found in part of alpha", m)


def calculate_frq_dep_alpha_beta(frq):
    rs.alpha_mat[:] = rs.alpha_mat_com[:]
    rs.beta_mat[:] = rs.beta_mat_com[:]

    for body1 in range(rs.nVessels):  # Influencing body

        for body2 in range(rs.nVessels):  # Influenced body

            print('Calculating frequency dependent influence of body ',
                  body1 + 1, 'on ', body2 + 1)

            # Influencing panel Q(xi,eta,zeta)
            for m in range(sh.sim.Vessels[body1].Grid.nUWPan):

                if body1 == 0:
                    indx1 = m
                else:
                    indx1 = m + rs.PanelRange[body1 - 1]

                if body2 == 0:
                    indx2a = 0
                else:
                    indx2a = rs.PanelRange[body2 - 1]
                indx2b = rs.PanelRange[body2]

                panP = sh.sim.Vessels[body2].Grid.UWPanels
                panQ = sh.sim.Vessels[body1].Grid.UWPanels

                panQ_Image = sh.sim.Vessels[body1].Grid.UWPanels_Image

                # int_dGbydn &int_G changes to int_G_dGbydn
                # int_G, int_dGbydn = int_G_dGbydn(panP, panQ, m, frq)

                # if np.any(np.isnan(int_dGbydn)):
                #     print("nan found in part of dGbydn", m)

                # rs.alpha_mat[indx2a:indx2b, indx1] += (1 / (2.0 * np.pi)) * \
                #     int_dGbydn

                # rs.beta_mat[indx2a:indx2b, indx1] += (1 / (4.0 * np.pi)) * \
                #     int_G

                if frq > 0.0:
                    # verify that green function takes a single scalar frequency and not a vector
                    integral_G, integral_dGdn = int_G_dGbydn(panP, panQ, m, frq)

                    rs.alpha_mat[indx2a:indx2b, indx1] += (1 / (2.0 * np.pi)) * integral_dGdn
                    for i in range(rs.nPan):
                        rs.alpha_mat[i, i] = - 1.0  # Modified on 03-Dec-2020

                    rs.beta_mat[indx2a:indx2b, indx1] += (1 / (4.0 * np.pi)) * integral_G

                else:
                    # Modified on 01-Mar-2021 for infinite frequency case
                    # Convert (1/r + 1/rp) to (1/r - 1/rp)

                    if frq < 0.0:

                        rs.alpha_mat[indx2a:indx2b, indx1] += (1 / (2.0 * np.pi)) * ( \
                                    -2.0 * int_dbydn_one_by_r(panP, panQ_Image, m))
                        for i in range(rs.nPan):
                            rs.alpha_mat[i, i] = - 1.0

                        rs.beta_mat[indx2a:indx2b, indx1] += (1 / (4.0 * np.pi)) * ( \
                                    -2.0 * int_one_by_r(panP, panQ_Image, m))


def calculate_vn(frq):
    if frq > 0:
        for body in range(rs.nVessels):

            if body == 0:
                indx1a = 0
            else:
                indx1a = rs.PanelRange[body - 1]
            indx1b = rs.PanelRange[body]

            for j in range(6):  # vn = j * wn * nk
                rs.vn[indx1a:indx1b, body * 6 + j] = 1j * \
                                                     frq * rs.n_terms[indx1a:indx1b, body * 6 + j]

        for i in range(rs.nbet):
            dphiI_by_dn = calculate_dphiI_by_dn(frq, rs.bet[i])
            rs.vn[:, 6 * rs.nVessels + i] = dphiI_by_dn
    else:
        for body in range(rs.nVessels):

            if body == 0:
                indx1a = 0
            else:
                indx1a = rs.PanelRange[body - 1]
            indx1b = rs.PanelRange[body]
        for j in range(6):
            rs.vn[indx1a:indx1b, body * 6 + j] = rs.n_terms[indx1a:indx1b, body * 6 + j]

    # Code snippet to save vn csv file for each frequency
    # rs.vn.tofile('vn.csv', sep=',')
    # np.savetxt('./vn'+str(frq)+'.csv', rs.vn, delimiter=",")


def calculate_dphiI_by_dn(frq, bet):
    # Change frq to omg
    # This needs to be updated to calculate dphiI_by_dn
    dphiI_by_dn = np.zeros(rs.nPan, dtype=complex)

    k = (frq ** 2) / (sh.g)

    # extracting unit normals of panels from n_terms
    gnx = rs.globnorm[:, 0]
    gny = rs.globnorm[:, 1]
    gnz = rs.globnorm[:, 2]

    # centroids of panels in global coordinates
    xc = rs.cen[:, 0]
    yc = rs.cen[:, 1]
    zc = rs.cen[:, 2]

    dphiI_by_dx = (frq * np.cos(bet) * np.exp(-1j * k *
                                              (xc * np.cos(bet) + yc * np.sin(bet))) * np.exp(k * zc))
    dphiI_by_dy = (frq * np.sin(bet) * np.exp(-1j * k *
                                              (xc * np.cos(bet) + yc * np.sin(bet))) * np.exp(k * zc))
    dphiI_by_dz = (1j * frq * np.exp(-1j * k * (xc * np.cos(bet) + yc * np.sin(bet))) * np.exp(k * zc))

    dphiI_by_dn = gnx * dphiI_by_dx + gny * dphiI_by_dy + gnz * dphiI_by_dz

    return dphiI_by_dn


def calculate_sigma():
    if np.any(np.isnan(rs.alpha_mat_com)):
        print("nan found in alpha")

    # if np.any(np.isnan(rs.vn)):
    #     print("nan found in vn")
    rs.LU, rs.piv, info = zgetrf(rs.alpha_mat)

    rs.sigma, info = zgetrs(rs.LU, rs.piv, rs.vn)

    rs.sigma = 2.0 * rs.sigma


def calculate_phi():
    # rs.phi = np.zeros((rs.nPan, 6 * rs.nVessels + rs.nbet), dtype=complex)
    rs.phi = rs.beta_mat @ rs.sigma


def calculate_phiI(frq):
    # centroids of panels in global coordinates
    xc = rs.cen[:, 0]
    yc = rs.cen[:, 1]
    zc = rs.cen[:, 2]
    k = (frq ** 2) / (sh.g)

    if frq > 0:
        for i in range(rs.nbet):
            rs.phiI[:, i] = 1j * sh.g * np.exp(-1j * k * (xc * np.cos(rs.bet[i]) + yc * np.sin(rs.bet[i]))) * np.exp(
                k * zc) / frq


def calculate_radiation(index):
    # Waves generated due to movement of ship
    # rs.phi = np.zeros((rs.nPan, 6 * rs.nVessels + rs.nbet), dtype=complex)

    for j in range(6 * rs.nVessels):
        for k in range(6 * rs.nVessels):
            rho = sh.rho
            body = np.floor_divide(j, 6)
            if body == 0:
                index1 = 0
                index2 = rs.PanelRange[0]

            else:
                index1 = rs.PanelRange[body - 1]
                index2 = rs.PanelRange[body]
            nj = rs.n_terms[index1:index2, j]

            ds = sh.sim.Vessels[body].Grid.UWPanels.Area
            if rs.frq[index] > 0.0:
                rs.AddMass[index, j, k] = -(rho / rs.frq[index]) * \
                                          np.sum(nj * rs.phi[index1:index2, k].imag * ds)
                rs.RadDamp[index, j, k] = -rho * \
                                          np.sum(nj * rs.phi[index1:index2, k].real * ds)
            else:
                rs.AddMass[index, j, k] = -(rho) * \
                                          np.sum(nj * rs.phi[index1:index2, k].real * ds)
                rs.RadDamp[index, j, k] = 0.0

    print("printing Added Mass-------------")
    print("addedmass size", rs.AddMass.shape)
    print(rs.AddMass / rho)
    a_file = open("test.txt", "w")
    for row in rs.AddMass:
        np.savetxt(a_file, row)

    a_file.close()


def calculate_scattering(index):
    # Diffraction forces, due to reflection of waves from ship
    # rs.ScatFrc = np.zeros((rs.nfrq, 6 * rs.nVessels, rs.nbet), dtype=complex)
    # From direct pressure integration
    # rs.phi = np.zeros((rs.nPan, 6 * rs.nVessels + rs.nbet), dtype=complex)
    # rs.n_terms = np.zeros((rs.nPan, 6 * rs.nVessels))
    # rs.frq : nfrq x 1
    # sh.sim.Vessels[body1].Grid.UWPanels.Area

    for k in range(6 * rs.nVessels):
        for j in range(rs.nbet):

            frq = rs.frq[index]
            rho = sh.rho
            body = np.floor_divide(k, 6)

            if body == 0:
                index1 = 0
                index2 = rs.PanelRange[0]

            else:
                index1 = rs.PanelRange[body - 1]
                index2 = rs.PanelRange[body]

            nk = rs.n_terms[index1:index2, k]

            ds = sh.sim.Vessels[body].Grid.UWPanels.Area

            rs.ScatFrc[index, k, j] = -1j * frq * rho * \
                                      np.sum(nk * rs.phi[index1:index2, 6 * rs.nVessels + j] * ds)
            # we need to calculate scattering force for one wave frequency since the driver function calculate_radition
            # runs like that.
            # rs.ScatFrc[index, i, j] = -1j * w * rho * np.sum(nk * calculate_phiI(rs.frq[index], rs.bet[j], index1, index2) * ds)
            # rs.ScatFrc[index, k, j] /= (rho*sh.g)
            rs.FKFrc[index, k, j] = -1j * frq * rho * np.sum(nk * rs.phiI[index1:index2, j] * ds)

if __name__ == '__main__':
    calculate_radiation_scattering()