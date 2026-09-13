import numpy as np

def ABCD_Series_C6(C6, fk):

    j = 1j
    w = 2 * np.pi * fk
    Y0 = j * w * C6
    Z0 = 1 / Y0

    A = 1
    B = Z0
    C = 0
    D = 1

    T = np.array([[A, B],
                  [C, D]])

    return T
