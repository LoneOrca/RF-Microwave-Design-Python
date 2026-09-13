import numpy as np

#-------------------------------
def ABCD_Shunt_C7(C7, fk):
#-------------------------------
# ABCD Matrix Shunt C
#-------------------------------

    j = 1j
    w = 2 * np.pi * fk

    if fk == 0:
        C = 0
    else:
        Z0 = j * w * C7
        y0 = 1 / Z0
        C = 1 / y0

    A = 1
    B = 0
    D = 1

    T = np.array([[A, B],
                  [C, D]])

    return T
