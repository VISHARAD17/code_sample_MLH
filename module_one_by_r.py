import numpy as np
import module_shared as sh
from class_panels import Panels
import matplotlib.pyplot as plt

# Will be used for calculation of alpha_mat
# Calculates the influence of panel Qind in panQ at panP_cen points
def int_dbydn_one_by_r(panP, panQ, Qind):
    panP_cen = panP.Centroids_Global
    
   # (1/r) case
    a = np.zeros(panP_cen.shape[0])
    tol = 1e-6
    qvert = panQ.Vertices_Local[Qind, :, :]
    #qcen = panQ.Centroids[0, :]
    gp = panP.Centroids_Global[:, :]
    gq = panQ.Centroids_Global[Qind,:]
    k = gp - gq
    nq = panQ.UnitVecs_Global[Qind, :, :]
    fp = (k.dot(nq)) # field centroid points in local co-ords of source
    # converting panelP centroids co-ords to local co-ords of panelQ
    #vx = np.zeros(pcen.shape[0])
    #vy = np.zeros(pcen.shape[0])
    #vz = np.zeros(pcen.shape[0])
    # Coords
    x,y,z = fp[:,0] , fp[:,1] , fp[:,2]
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    print("X:\n", x)
    print("\nY:\n", y)
    print("\nZ:\n", z)
    x1, x2, x3, x4 = qvert[0, 0], qvert[1, 0], qvert[2, 0], qvert[3, 0]
    y1, y2, y3, y4 = qvert[0, 1], qvert[1, 1], qvert[2, 1], qvert[3, 1]
    d12 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    d23 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    d34 = np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2)
    d41 = np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)
    m12 = (y2 - y1) / (x2 - x1)
    m23 = (y3 - y2) / (x3 - x2)
    m34 = (y4 - y3) / (x4 - x3)
    m41 = (y1 - y4) / (x1 - x4)
    # m1 = (y2 - y1) / (x2 - x1)
    # m12 = np.where(m1==np.nan,0,m1)
    # m2 = (y3 - y2) / (x3 - x2)
    # m23 = np.where(m2 == np.nan, 0, m2)
    # m3 = (y4 - y3) / (x4 - x3)
    # m34 = np.where(m3 == np.nan, 0, m3)
    # m4 = (y1 - y4) / (x1 - x4)
    # m41 = np.where(m4 == np.nan, 0, m4)
    # if (x2-x1) != 0:
    #     m12 = (y2 - y1) / (x2 - x1)
    # else:
    #     m12 = 0
    # if (x3-x2) != 0:
    #     m23 = (y3 - y2) / (x3 - x2)
    # else:
    #     m23 = 0
    # if (x4-x3) != 0:
    #     m34 = (y4 - y3) / (x4 - x3)
    # else:
    #     m34 = 0
    # if (x1-x4) != 0:
    #     m41 = (y1 - y4) / (x1 - x4)
    # else:
    #     m41 = 0
    r = np.zeros((panP_cen.shape[0],4))
    r[:, 0] = np.sqrt((x-x1)**2 + (y - y1)**2+ z**2)
    r[:, 1] = np.sqrt((x-x2)**2 + (y - y2)**2+ z**2)
    r[:, 2] = np.sqrt((x-x3)**2 + (y - y3)**2+ z**2)
    r[:, 3] = np.sqrt((x-x4)**2 + (y - y4)**2+ z**2)
    print("\nR:\n",r)
    e = np.zeros((panP_cen.shape[0], 4))
    e[:, 0] = z ** 2 + (x - x1)**2
    e[:, 1] = z ** 2 + (x - x2) ** 2
    e[:, 2] = z ** 2 + (x - x3) ** 2
    e[:, 3] = z ** 2 + (x - x4) ** 2
    h = np.zeros((panP_cen.shape[0],4))
    h[:,0] = (y - y1) * (x - x1)
    h[:, 1] = (y - y2) * (x - x2)
    h[:, 2] = (y - y3) * (x - x3)
    h[:, 3] = (y - y4) * (x - x4)
    vx = np.where(d12>tol,((y2 - y1)/d12)*np.log((r[:,0]+r[:,1]-d12)/(r[:,0]+r[:,1]+d12)),0) + \
            np.where(d23>tol,((y3 - y2)/d23)*np.log((r[:,1]+r[:,2]-d23)/(r[:,1]+r[:,2]+d23)),0) + \
            np.where(d34>tol,((y4 - y3)/d34)*np.log((r[:,2]+r[:,3]-d34)/(r[:,2]+r[:,3]+d34)),0) + \
            np.where(d41>tol,((y1 - y4)/d41)*np.log((r[:,3]+r[:,0]-d41)/(r[:,3]+r[:,0]+d41)),0)
    print("\nVx\n:",vx)
    vy = np.where(d12>tol,((x1 - x2)/d12)*np.log((r[:,0]+r[:,1]-d12)/(r[:,0]+r[:,1]+d12)),0) + \
            np.where(d23>tol,((x2 - x3)/d23)*np.log((r[:,1]+r[:,2]-d23)/(r[:,1]+r[:,2]+d23)),0) + \
            np.where(d34>tol,((x3 - x4)/d34)*np.log((r[:,2]+r[:,3]-d34)/(r[:,2]+r[:,3]+d34)),0) + \
            np.where(d41>tol,((x4 - x1)/d41)*np.log((r[:,3]+r[:,0]-d41)/(r[:,3]+r[:,0]+d41)),0)
    print("\nVy\n:", vy)
    if np.any(np.isnan([m12,m23,m34,m41])):
        vz = 0
    else:
        vz = np.where(np.abs(z) > tol,np.arctan((m12 * e[:,0] - h[:,0]) / (z * r[:,0])) - np.arctan((m12 * e[:,1] - h[:,1]) / (z * r[:,1])),0.0) + \
                np.where(np.abs(z) > tol, np.arctan((m23 * e[:,1] - h[:,1]) / (z * r[:,1])) - np.arctan((m23 * e[:,2] - h[:,2]) / (z * r[:,2])),0.0) + \
                np.where(np.abs(z) > tol, np.arctan((m34 * e[:,2] - h[:,2]) / (z * r[:,2])) - np.arctan((m34 * e[:,3] - h[:,3]) / (z * r[:,3])),0.0) + \
                np.where(np.abs(z) > tol, np.arctan((m41 * e[:,3] - h[:,3]) / (z * r[:,3])) - np.arctan((m41 * e[:,0] - h[:,0]) / (z * r[:,0])),0.0)
    print("\nVz\n:", vz)
    #norm = np.zeros(panP_cen.shape[0])
    #norm = panP.UnitVecs_Global.dot(panQ.UnitVecs_Global[Qind])
    npq = panP.UnitVecs_Global[:, 2, :].dot(panQ.UnitVecs_Global[0, :, :])
    nx = npq[:,0] # panP.UnitVecs_Global[:,0,:].dot(panQ.UnitVecs_Global[Qind, :, 0])
    ny = npq[:,1] # panP.UnitVecs_Global[:,1,:].dot(panQ.UnitVecs_Global[Qind, :, 1])
    nz = npq[:,2] # panP.UnitVecs_Global[:,2,:].dot(panQ.UnitVecs_Global[Qind, :, 2])
    a = vx * nx + vy * ny + vz * nz
    print("\nDbyDn-a:\n",a)

    return a

# Will be used for calculation of beta_mat
# Calculates the influence of panel Qind in panQ at panP_cen points
def int_one_by_r(panP, panQ, Qind):
    panP_cen = panP.Centroids_Global
    # (1/r) case
    # b = np.zeros(panP_cen.shape[0])
    tol = 1e-6
    qvert = panQ.Vertices_Local[Qind, :, :]
    # qcen = panQ.Centroids[0, :]
    gp = panP.Centroids_Global[:, :]
    gq = panQ.Centroids_Global[Qind, :]
    k = gp - gq
    nq = panQ.UnitVecs_Global[Qind, :, :]
    fp = k.dot(nq)  # field centroid points in local co-ords of source
    # converting panelP centroids co-ords to local co-ords of panelQ
    # vx = np.zeros(pcen.shape[0])
    # vy = np.zeros(pcen.shape[0])
    # vz = np.zeros(pcen.shape[0])
    # Coords
    x, y, z = fp[:, 0], fp[:, 1], fp[:, 2]
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    x1, x2, x3, x4 = qvert[0, 0], qvert[1, 0], qvert[2, 0], qvert[3, 0]
    y1, y2, y3, y4 = qvert[0, 1], qvert[1, 1], qvert[2, 1], qvert[3, 1]
    d12 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    d23 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    d34 = np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2)
    d41 = np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)
    m12 = (y2 - y1) / (x2 - x1)
    m23 = (y3 - y2) / (x3 - x2)
    m34 = (y4 - y3) / (x4 - x3)
    m41 = (y1 - y4) / (x1 - x4)
    # m1 = (y2 - y1) / (x2 - x1)
    # m12 = np.where(m1 == np.nan, 0, m1)
    # m2 = (y3 - y2) / (x3 - x2)
    # m23 = np.where(m2 == np.nan, 0, m2)
    # m3 = (y4 - y3) / (x4 - x3)
    # m34 = np.where(m3 == np.nan, 0, m3)
    # m4 = (y1 - y4) / (x1 - x4)
    # m41 = np.where(m4 == np.nan, 0, m4)
    # if (x2-x1) != 0:
    #     m12 = (y2 - y1) / (x2 - x1)
    # else:
    #     m12 = 0
    # if (x3-x2) != 0:
    #     m23 = (y3 - y2) / (x3 - x2)
    # else:
    #     m23 = 0
    # if (x4-x3) != 0:
    #     m34 = (y4 - y3) / (x4 - x3)
    # else:
    #     m34 = 0
    # if (x1-x4) != 0:
    #     m41 = (y1 - y4) / (x1 - x4)
    # else:
    #     m41 = 0
    r = np.zeros((panP_cen.shape[0], 4))
    r[:, 0] = np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + z ** 2)
    r[:, 1] = np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + z ** 2)
    r[:, 2] = np.sqrt((x - x3) ** 2 + (y - y3) ** 2 + z ** 2)
    r[:, 3] = np.sqrt((x - x4) ** 2 + (y - y4) ** 2 + z ** 2)
    e = np.zeros((panP_cen.shape[0], 4))
    e[:, 0] = z ** 2 + (x - x1) ** 2
    e[:, 1] = z ** 2 + (x - x2) ** 2
    e[:, 2] = z ** 2 + (x - x3) ** 2
    e[:, 3] = z ** 2 + (x - x4) ** 2
    h = np.zeros((panP_cen.shape[0], 4))
    h[:, 0] = (y - y1) * (x - x1)
    h[:, 1] = (y - y2) * (x - x2)
    h[:, 2] = (y - y3) * (x - x3)
    h[:, 3] = (y - y4) * (x - x4)

    bx = np.where(d12>tol,(((x-x1)*(y2 - y1)-(y - y1)*(x2 - x1))/d12)*np.log((r[:,0]+r[:,1]+d12)/(r[:,0]+r[:,1]-d12)),0) +\
     np.where(d23>tol,(((x-x2)*(y3 - y2)-(y - y2)*(x3 - x2))/d23)*np.log((r[:,1]+r[:,2]+d23)/(r[:,1]+r[:,2]-d23)),0) +\
     np.where(d34>tol,(((x-x3)*(y4 - y3)-(y - y3)*(x4 - x3))/d34)*np.log((r[:,2]+r[:,3]+d34)/(r[:,2]+r[:,3]-d34)),0) +\
     np.where(d41>tol,(((x-x4)*(y1 - y4)-(y - y4)*(x1 - x4))/d41)*np.log((r[:,3]+r[:,0]+d41)/(r[:,3]+r[:,0]-d41)),0)
    if np.any(np.isnan([m12,m23,m34,m41])):
        by = 0
    else:
        by = np.where(np.abs(z) > tol, z*((np.arctan((m12*e[:,0]-h[:,0])/(z*r[:,0]))-np.arctan((m12*e[:,1]-h[:,1])/(z*r[:,1])))+
                                                  (np.arctan((m23*e[:,1]-h[:,1])/(z*r[:,1]))-np.arctan((m23*e[:,2]-h[:,2])/(z*r[:,2])))+
                                                  (np.arctan((m34*e[:,2]-h[:,2])/(z*r[:,2]))-np.arctan((m34*e[:,3]-h[:,3])/(z*r[:,3])))+
                                                  (np.arctan((m41*e[:,3]-h[:,3])/(z*r[:,3]))-np.arctan((m41*e[:,0]-h[:,0])/(z*r[:,0])))),0.0)
    b = bx-by


    return b

def int_dbydn_one_by_r_desingular(panP, panQ, Qind):
    gxp = panP.Centroids_Global[:, 0]
    gyp = panP.Centroids_Global[:, 1]
    gzp = panP.Centroids_Global[:, 2]

    gxq = panQ.Centroids_Global[Qind, 0]
    gyq = panQ.Centroids_Global[Qind, 1]
    gzq = panQ.Centroids_Global[Qind, 2]

    x = (gxp - gxq) * panQ.UnitVecs_Global[Qind, 0, 0] \
        + (gyp - gyq) * panQ.UnitVecs_Global[Qind, 0, 1] \
        + (gzp - gzq) * panQ.UnitVecs_Global[Qind, 0, 2]

    y = (gxp - gxq) * panQ.UnitVecs_Global[Qind, 1, 0] \
        + (gyp - gyq) * panQ.UnitVecs_Global[Qind, 1, 1] \
        + (gzp - gzq) * panQ.UnitVecs_Global[Qind, 1, 2]

    z = (gxp - gxq) * panQ.UnitVecs_Global[Qind, 2, 0] \
        + (gyp - gyq) * panQ.UnitVecs_Global[Qind, 2, 1] \
        + (gzp - gzq) * panQ.UnitVecs_Global[Qind, 2, 2]

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    vx = x / (r ** 3)
    vy = y / (r ** 3)
    vz = z / (r ** 3)

    nx = np.sum(panP.UnitVecs_Global[:, 2, :] * panQ.UnitVecs_Global[Qind, 0, :], axis=1)
    ny = np.sum(panP.UnitVecs_Global[:, 2, :] * panQ.UnitVecs_Global[Qind, 1, :], axis=1)
    nz = np.sum(panP.UnitVecs_Global[:, 2, :] * panQ.UnitVecs_Global[Qind, 2, :], axis=1)

    vel_theory = np.where(r < 1.0e-6, 0.0, vx * nx + vy * ny + vz * nz)

    return vel_theory*panQ.Area[Qind]

def int_one_by_r_desingular(panP, panQ, Qind):
    gxp = panP.Centroids_Global[:, 0]
    gyp = panP.Centroids_Global[:, 1]
    gzp = panP.Centroids_Global[:, 2]

    gxq = panQ.Centroids_Global[Qind, 0]
    gyq = panQ.Centroids_Global[Qind, 1]
    gzq = panQ.Centroids_Global[Qind, 2]

    x = (gxp - gxq) * panQ.UnitVecs_Global[Qind, 0, 0] \
        + (gyp - gyq) * panQ.UnitVecs_Global[Qind, 0, 1] \
        + (gzp - gzq) * panQ.UnitVecs_Global[Qind, 0, 2]

    y = (gxp - gxq) * panQ.UnitVecs_Global[Qind, 1, 0] \
        + (gyp - gyq) * panQ.UnitVecs_Global[Qind, 1, 1] \
        + (gzp - gzq) * panQ.UnitVecs_Global[Qind, 1, 2]

    z = (gxp - gxq) * panQ.UnitVecs_Global[Qind, 2, 0] \
        + (gyp - gyq) * panQ.UnitVecs_Global[Qind, 2, 1] \
        + (gzp - gzq) * panQ.UnitVecs_Global[Qind, 2, 2]

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    pot_theory = np.where(r < 1.0e-6, 0.0, 1 / r)

    return pot_theory*panQ.Area[Qind]

def calculate_normal_panP_panels(n,R):
    tol = 1.0e-3
    coord = np.zeros((np.shape(n)[0],4,3))

    for i in range(np.shape(n)[0]):
        if np.abs(np.sqrt(n[i,0]**2 + n[i,1]**2 + n[i,2]**2) - 1.0) > tol:
            print('ERROR: Directions are expected to be unit normals')
            print(n[i,:], np.sqrt(n[i,0]**2 + n[i,1]**2 + n[i,2]**2))
            exit()

        theta = np.arctan2(-n[i, 0], n[i, 1])
        coord[i, 0, :] = np.array([np.cos(theta), np.sin(theta), 0.0])
        coord[i, 1, :] = np.cross(n[i, :], coord[i, 0, :])
        coord[i, 2, :] = -coord[i, 0, :]
        coord[i, 3, :] = -coord[i, 1, :]

        coord[i, :, 0] = coord[i, :, 0] + R[i] * n[i, 0]
        coord[i, :, 1] = coord[i, :, 1] + R[i] * n[i, 1]
        coord[i, :, 2] = coord[i, :, 2] + R[i] * n[i, 2]

    return coord



if __name__ == "__main__":
    print('Debug mode for module_one_by_r')

    eps = 1.0e-3

    npanP = 200
    npanQ = 1

    Qind = 0

    panP_coord = np.zeros((npanP, 4, 3))
    panQ_coord = np.zeros((npanQ, 4, 3))

    # panQ_coord = 2 * np.random.random(size=(npanQ,4,3)) - 1

    panQ_coord[0, 0, :] = np.array([1.0, 0.0, 0.0])
    panQ_coord[0, 1, :] = np.array([0.0, 1.0, 0.0])
    panQ_coord[0, 2, :] = np.array([-1.0, 0.0, 0.0])
    panQ_coord[0, 3, :] = np.array([0.0, -1.0, 0.0])

    xbody = np.zeros(4)
    panQ = Panels(panQ_coord, npanQ, xbody)

    Radius = np.ones((npanP))
    dirs = np.zeros((npanP,3))

    # theta = np.linspace(0,4*np.pi,num=npanP)
    # theta = np.ones((npanP)) * (135 * np.pi / 180)
    # phi = np.ones((npanP)) * (30 * np.pi / 180)
    np.random.seed(0)
    theta = np.random.random(size=(npanP)) * (2.0*np.pi)
    phi = np.random.random(size=(npanP)) * (np.pi) - np.pi/2.0

    # theta_val = np.random.random() * (2 * np.pi)
    # phi_val = np.random.random() * (np.pi) - np.pi/2

    # theta_val = 45 * np.pi / 180
    # phi_val = 0 * np.pi / 180

    # theta = np.ones((npanP)) * theta_val
    # phi = np.ones((npanP)) * phi_val

    # print('Theta = ', theta_val*180/np.pi)
    # print('Phi = ', phi_val*180/np.pi)

    for i in range(npanP):
        dirs[i,:] = np.array([np.cos(theta[i]) * np.cos(phi[i]),
                              np.sin(theta[i]) * np.cos(phi[i]),
                              np.sin(phi[i])])
        Radius[i] = 0.1*i + 1.0

    panP_coord = calculate_normal_panP_panels(dirs, Radius)
    panP = Panels(panP_coord, npanP, xbody)

    dist = np.linalg.norm(panP.Centroids_Global - panQ.Centroids_Global, axis=1)

    vel_calc = int_dbydn_one_by_r(panP, panQ, Qind)
    pot_calc = int_one_by_r(panP, panQ, Qind)

    vel_theory = int_dbydn_one_by_r_desingular(panP, panQ, Qind)
    pot_theory = int_one_by_r_desingular(panP, panQ, Qind)

    # print('Potentials:')
    # print('Theory:', pot_theory)
    # print('Calculated: ', pot_calc)
    #
    # print('Velocities:')
    # print('Theory:', vel_theory)
    # print('Calculated: ', vel_calc)

    plt.figure(1)
    plt.semilogy(dist,abs(vel_theory),dist,abs(vel_calc),'--')
    plt.title('Velocity due to source')
    plt.ylabel('Velocity')
    plt.xlabel('Distance in m')
    plt.legend(['Point Source', 'Panel Source'])
    #plt.savefig('Velocity_one_by_r.png')
    plt.show()

    plt.figure(2)
    plt.semilogy(dist, abs(pot_theory), dist, abs(pot_calc), '--')
    plt.title('Potential due to source')
    plt.ylabel('Potential')
    plt.xlabel('Distance in m')
    plt.legend(['Point Source', 'Panel Source'])
    #plt.savefig('Potential_one_by_r.png')
    plt.show()