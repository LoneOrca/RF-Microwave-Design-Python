import numpy as np
from CParam_Interp import CParam_Interp

def S_Param_Interp(S, freq, fx):

    # Find fx in the frequency array
    Ix = np.where(freq == fx)[0]

    Sx = np.zeros((2, 2), dtype=complex)

    # If fx is already present
    if len(Ix) != 0:
        Sx[:, :] = S[Ix[0], :, :]
        return Sx

    # If fx is not present, interpolate each S-parameter
    for ii in range(2):
        for jj in range(2):
            Sx[ii, jj] = CParam_Interp(
                S[:, ii, jj], freq, fx
            )

    return Sx
