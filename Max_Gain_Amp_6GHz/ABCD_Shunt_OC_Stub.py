import numpy as np

def ABCD_Shunt_OC_Stub(Z0x, theta_x):

    Z = -1j * Z0x / np.tan(np.deg2rad(theta_x))
    Y = 1 / Z

    A = 1
    B = 0
    C = Y
    D = 1

    T = np.array([
        [A, B],
        [C, D]
    ], dtype=complex)

    return T
