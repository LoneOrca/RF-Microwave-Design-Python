import numpy as np

def phase_d(Z):

    X = np.real(Z)
    Y = np.imag(Z)

    theta_1 = np.degrees(np.arctan2(Y, X))
    theta_2 = np.mod(theta_1, 360)

    return theta_1, theta_2
