import numpy as np

def ABCD_Shunt_C1(C1, fk):

    j = 1j
    w = 2 * np.pi * fk

    if fk == 0:
        C = 0
    else:
        Z0 = j * w * C1
        Y0 = 1 / Z0
        C = 1 / Y0

    A = 1
    B = 0
    D = 1

    T = np.array([[A, B],
                  [C, D]])

    return T
