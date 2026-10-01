import numpy as np

def Param_Interp(P, freq, fx):

    # Find fx in the frequency array
    Ix = np.where(freq == fx)[0]

    # If fx is already present
    if len(Ix) != 0:
        Px = P[Ix[0]]
        return Px

    # Frequency below fx
    f1 = freq[freq < fx][-1]

    I1 = freq == f1
    P1 = P[I1][0]

    # Frequency above fx
    f2 = freq[freq > fx][0]

    I2 = freq == f2
    P2 = P[I2][0]

    # Linear interpolation
    m = (P2 - P1) / (f2 - f1)

    Px = P1 + m * (fx - f1)

    return Px
