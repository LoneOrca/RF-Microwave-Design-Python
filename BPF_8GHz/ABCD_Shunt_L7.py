import numpy as np

def ABCD_Shunt_L7(L7, fk):

    j = 1j
    w = 2 * np.pi * fk
    Z0 = j * w * L7
    Y0 = 1 / Z0

    A = 1
    B = 0
    C = Y0
    D = 1

    T = np.array([[A, B],
                  [C, D]])

    return T
