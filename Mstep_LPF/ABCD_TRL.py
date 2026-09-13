import numpy as np

def cosd(x):
    return np.cos(x * np.pi / 180)

def sind(x):
    return np.sin(x * np.pi / 180)


def ABCD_TRL(Z0x, theta_x):

    j = 1j

    K = cosd(theta_x)
    L = sind(theta_x)

    Y0x = 1 / Z0x

    A = K
    B = j * Z0x * L
    C = j * Y0x * L
    D = K

    T = np.array([[A, B],
                  [C, D]])

    return T
