import numpy as np

def Polar_2_Rect(R, theta):

    # theta is in degrees
    Z = R * (np.cos(np.deg2rad(theta)) + 1j * np.sin(np.deg2rad(theta)))

    return Z
