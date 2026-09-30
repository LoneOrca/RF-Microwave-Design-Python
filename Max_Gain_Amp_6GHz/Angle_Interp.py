import numpy as np


def Angle_Interp(P, freq, fx):
    
    P = np.asarray(P)
    freq = np.asarray(freq)

    Ix = np.where(freq == fx)[0]

    IQ1 = 1
    IQ2 = 1

    if len(Ix) != 0:
        Px = P[Ix[0]]
        return Px

    f1 = freq[freq < fx][-1]
    I1 = freq == f1
    P1 = P[I1][0]

    if P1 > 90 and P1 < 180:
        IQ1 = 2

    if P1 < -90 and P1 > -180:
        IQ1 = 3

    f2 = freq[freq > fx][0]
    I2 = freq == f2
    P2 = P[I2][0]

    if P2 > 90 and P2 < 180:
        IQ2 = 2

    if P2 < -90 and P2 > -180:
        IQ2 = 3

    if IQ1 * IQ2 == 6 and IQ1 == 3:
        P1 = 360 + P1

    if IQ1 * IQ2 == 6 and IQ2 == 3:
        P2 = 360 + P2

    m = (P2 - P1) / (f2 - f1)

    Px = P1 + m * (fx - f1)

    return Px
