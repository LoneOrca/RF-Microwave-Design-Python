import numpy as np

def S_to_ABCD(S, Z0=50):

    # If Z0 is a single value, use it for both ports
    Z0 = np.atleast_1d(Z0).astype(float)

    if len(Z0) == 1:
        Z0 = np.array([Z0[0], Z0[0]])

    Zm = np.sqrt(Z0[0] * Z0[1])

    S11 = S[0, 0]
    S12 = S[0, 1]
    S21 = S[1, 0]
    S22 = S[1, 1]

    Den = 2 * Zm * S21

    A_Num = Z0[0] * ((1 + S11) * (1 - S22) + S12 * S21)
    A = A_Num / Den

    B_Num = np.prod(Z0) * ((1 + S11) * (1 + S22) - S12 * S21)
    B = B_Num / Den

    C_Num = ((1 - S11) * (1 - S22) - S12 * S21)
    C = C_Num / Den

    D_Num = Z0[1] * ((1 - S11) * (1 + S22) + S12 * S21)
    D = D_Num / Den

    T = np.array([
        [A, B],
        [C, D]
    ], dtype=complex)

    return T
