import numpy as np
#------------------------
def ABCD_Series_L6(L6,fk):
#------------------------
# ABCD Matrix Series L
#------------------------
 j = 1j
 w = 2*np.pi*fk
 z0 = j*w*L6
 
 A = 1
 B = z0
 C = 0
 D = 1
 
 T = np.array([[A, B],
              [C, D]])
 return T
 

