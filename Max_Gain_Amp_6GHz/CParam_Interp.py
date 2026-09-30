import numpy as np

from phase_d import phase_d
from Param_Interp import Param_Interp
from Angle_Interp import Angle_Interp
from Polar_2_Rect import Polar_2_Rect


def CParam_Interp(P, freq, fx):

    P_Mag = np.abs(P)

    P_Ang, _ = phase_d(P)

    Px_Mag = Param_Interp(P_Mag, freq, fx)

    Px_Ang = Angle_Interp(P_Ang, freq, fx)

    Px = Polar_2_Rect(Px_Mag, Px_Ang)

    return Px
