import numpy as np

def ABCD_to_S(T, Z0):

    if len(Z0) == 1:
        Z0 = [Z0[0], Z0[0]]

    Z01 = Z0[0]
    Z02 = Z0[1]
    Zm = np.sqrt(Z01 * Z02)

    #--------------------------------

    A = T[0, 0]
    B = T[0, 1]
    C = T[1, 0]
    D = T[1, 1]

    delta = np.linalg.det(T)

    #--------------------------------

    S = np.zeros((2, 2), dtype=complex)

    #--------------------------------

    if Z01 == 0:
        S[1, 1] = (-A * Z02 + B) / (A * Z02 + B)
        return S

    #--------------------------------

    if Z01 == np.inf:
        S[1, 1] = (-D - C * Z02) / (D + C * Z02)
        return S

    #--------------------------------

    if Z02 == 0:
        S[0, 0] = (-D * Z01 + B) / (D * Z01 + B)
        return S

    #--------------------------------

    if Z02 == np.inf:
        S[0, 0] = (A - C * Z01) / (A + C * Z01)
        return S

    #--------------------------------

    Den = A * Z02 + D * Z01 + B + C * Z01 * Z02

    Num_11 = A * Z02 - D * Z01 + B - C * Z01 * Z02
    Num_21 = 2 * Zm
    Num_12 = 2 * Zm * delta
    Num_22 = -A * Z02 + D * Z01 + B - C * Z01 * Z02

    #--------------------------------

    S[0, 0] = Num_11 / Den
    S[0, 1] = Num_12 / Den
    S[1, 0] = Num_21 / Den
    S[1, 1] = Num_22 / Den

    #--------------------------------

    if abs(C) == np.inf:
        S = np.array([[-1, 0],
                      [0, -1]])

    #--------------------------------

    if abs(B) == np.inf:
        S = np.array([[1, 0],
                      [0, 1]])

    return S
