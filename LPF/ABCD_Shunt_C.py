import numpy as np
#-------------------------------
def ABCD_Shunt_C(C,fk):
#-------------------------------
# ABCD Matrix Shunt C
#------------------------------
 j = 1j
 w = 2*np.pi*fk
 Z0 = j*w*C
 Y0=1/Z0

 A = 1
 B = 0
 C = 1/Y0
 D = 1

 T = np.array([[A, B],
              [C, D]])

 return T
  
 
