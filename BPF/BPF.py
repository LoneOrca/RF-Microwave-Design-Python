import numpy as np
from ABCD_Shunt_C1 import ABCD_Shunt_C1
from ABCD_Shunt_L1 import ABCD_Shunt_L1
from ABCD_Series_L2 import ABCD_Series_L2
from ABCD_Series_C2 import ABCD_Series_C2
from ABCD_Shunt_C3 import ABCD_Shunt_C3
from ABCD_Shunt_L3 import ABCD_Shunt_L3
from ABCD_to_S import ABCD_to_S
import scipy
import matplotlib.pyplot as plt

#--------Srinivas Nittala--------
# Lumped Element Band Pass Filter Design Matlab to Python transition code

G = 10**9
Meg = 10**6
k = 10**3
c = 10**-2
C = 3e8      #Speed of Light
m = 10**-3
u = 10**-6
n = 10**-9
p = 10**-12
IFigure = 0
NF = 32
f_min = 1.6*G
f_max = 2.4*G
df = 5*Meg
j = 1j
#-------------part-1---------------
fl = 1.9025*G
fh = 2.1025*G
f0 = np.sqrt(fl*fh)
dl = (fh - fl)/f0
Bwf = f0*dl
Bwf1 = Bwf*k
#-------------part-2---------------
IL_max = 0.75
S21_dB = -(IL_max)
S21_IL_min = 10**(S21_dB/20)
S11_IL_max = np.sqrt(1-(S21_IL_min)**2)
S11_IL_dB = 20*np.log10(S11_IL_max)
#-------------part-3---------------
RL_min = 13.4672
S11_dB = -(RL_min)
S11_RL_max = 10**(S11_dB/20)
S21_RL_min = np.sqrt(1-(S11_RL_max)**2)
S21_dB_min = 20*np.log10(S21_RL_min)
#-------------part-4---------------
Ap = -(S21_dB_min)
#Apl = 0.02 
#-------------part-5---------------
fls = 1.6396*G
fhs = 2.3224*G
fls_Il = 30
Als = fls_Il
fhs_Il = 20
Ahs = fhs_Il
#-------------part-6---------------
Ap1 = 10**(Ap/10)
Xp = Ap1 - 1
E = np.sqrt(Xp)
#Epsilon = np.ceil(E*100)/100
#-------------part-7---------------
Als1 = 10**(Als/10)
Xls = Als1 - 1
root_Xls = np.sqrt(Xls)
#-------------part-8---------------
Omega_ls = (1/dl)*((fls/f0)-(f0/fls))
xy = np.abs(Omega_ls)
y = xy - 1
Nls = np.arccosh((1/E)*(root_Xls))/np.arccosh(xy)
#-------------part-9---------------
Ahs1 = 10**(Ahs/10)
Xhs = Ahs1 - 1
root_Xhs = np.sqrt(Xhs)
Omega_hs = (1/dl)*((fhs/f0)-(f0/fhs))
Nhs = np.arccosh((1/E)*(root_Xhs))/np.arccosh(Omega_hs)
#-------------part-10---------------
N1 = int(np.ceil(Nls))
if N1 % 2 == 0:
  N1 += 1
Nc = N1

N2 = int(np.ceil(Nhs))
if N2 % 2 == 0:
  N2 += 1
Nd = N2

if N1 == N2:
    N = N1
    print("FILTER ORDER =", N)
else:
    print("N1 and N2 are different")

#-------------part-11---------------
g0 = 1.0000
g1 = 1.2275
g2 = 1.1525
g3 = 1.2275
g4 = 1.0000 
#-------------part-12---------------
Z0 = 50
Y0 = 1/Z0
w0 = 2*np.pi*f0
#-------------part-13---------------
#-----------Components--------------
#-----------Branch 1----------------
C1 = Y0*g1*(1/(w0*dl))
L1 = 1/((w0**2)*C1)
#-----------Branch 2----------------
L2 = (1/(w0*dl)*g2*Z0)
C2 = 1/((w0**2)*L2)
#-----------Branch 3----------------
C3 = Y0*g3*(1/(w0*dl))
L3 = 1/((w0**2)*C3)
#-------------part-14---------------
freq = np.arange(f_min, f_max + df, df) #Create Frequency Range
N_Freq = len(freq) #Count frequencies
S_Filter = np.zeros((N_Freq, 2, 2), dtype=complex) #Create apce for S param
for kk in range(N_Freq): #Loop through every frequency
 fk = freq[kk] #Pick Current Frequency
 T0 = np.eye(2)
 T1 = ABCD_Shunt_C1(C1,fk)
 T2 = ABCD_Shunt_L1(L1, fk)
 T3 = ABCD_Series_L2(L2,fk)
 T4 = ABCD_Series_C2(C2, fk)
 T5 = ABCD_Shunt_C3(C3,fk)
 T6 = ABCD_Shunt_L3(L3, fk)
 T  = T0 @ T1 @ T2 @ T3 @ T4 @ T5 @ T6
 S_Filter[kk, :, :] = ABCD_to_S(T, [Z0, Z0])
#-------------part-15---------------
S11_Filter = S_Filter[:, 0, 0]
S11_Filter_Mag = np.abs(S11_Filter)
S11_Filter_dB = 20 * np.log10(S11_Filter_Mag)

S21_Filter = S_Filter[:, 1, 0]
S21_Filter_Mag = np.abs(S21_Filter)
S21_Filter_dB = 20 * np.log10(S21_Filter_Mag)
#-------------part-16---------------
plt.figure()

plt.plot(freq / G, S11_Filter_dB, 'g',  linewidth = 6)

plt.xlabel("Frequency (GHz)")
plt.ylabel("S11 (dB)")
plt.title("S11 Lumped Element Band Pass Filter")

plt.xlim(1.6, 2.4)
plt.ylim(-30, 0)

plt.grid(True)
plt.show()
#------------
plt.figure()

plt.plot(freq / G, S21_Filter_dB, 'r', linewidth = 6)

plt.xlabel("Frequency (GHz)")
plt.ylabel("S21 (dB)")
plt.title("S21 Lumped Element Band Pass Filter")

plt.xlim(1.85, 2.15)
plt.ylim(-0.3, 0)

plt.grid(True)
plt.show()
#------------
plt.figure()

plt.plot(freq / G, S11_Filter_dB, 'g',  linewidth = 3)
plt.plot(freq / G, S21_Filter_dB, 'r', linewidth = 3)
plt.xlabel("Frequency (GHz)")
plt.ylabel("S-Parameters (dB)")
plt.title("S11 and S21 of Lumped element BPF")

plt.xlim(1.6, 2.4)
plt.ylim(-40, 0)

plt.grid(True)
plt.show()
#-------------part-17---------------
#---------Coupled Line-------------
#----Section 01----------
j01 = np.sqrt((((1/2)*np.pi)*dl)/(g0*g1))
Z0e_01 = Z0*(1+j01+(j01)**2)
Z0o_01 = Z0*(1-j01+(j01)**2)
#----Section 12----------
j12 = ((((0.5)*np.pi)*dl)/np.sqrt(g1*g2))
Z0e_12 = Z0*(1+j12+(j12)**2)
Z0o_12 = Z0*(1-j12+(j12)**2)
#----Section 23----------
j23 = ((((0.5)*np.pi)*dl)/np.sqrt(g2*g3))
Z0e_23 = Z0*(1+j23+(j23)**2)
Z0o_23 = Z0*(1-j23+(j23)**2)
#----Section 34----------
j34 = np.sqrt((((1/2)*np.pi)*dl)/(g3*g4))
Z0e_34 = Z0*(1+j34+(j34)**2)
Z0o_34 = Z0*(1-j34+(j34)**2)
print("--------part-1-------------")
print(f"f0 = {f0/G:.4f} GHz")
print(f"dl = {dl:.4f}")
print(f"Bwf = {Bwf/G:.4f} GHz")
print(f"Bwf1 = {Bwf1/G:.4f} MHz")
# part-1 *****GOOD*****
print("--------part-2-------------")
print(f"S21_dB = {S21_dB:.4f} dB")
print(f"S21_IL_min = {S21_IL_min:.4f}")
print(f"S11_IL_max = {S11_IL_max:.4f} W/W")
print(f"S11_IL_dB = {S11_IL_dB:.4f} dB")
# part-2 *****GOOD*****
print("--------part-3-------------")
print(f"S11_dB = {S11_dB:.4f} dB")
print(f"S11_RL_max = {S11_RL_max:.4f}")
print(f"S21_RL_min = {S21_RL_min:.4f} W/W")
print(f"S21_dB_min = {S21_dB_min:.4f} dB")
# part-3 *****GOOD*****
print("--------part-4-------------")
print(f"Ap = {Ap:.4f} dB")
print("Ripple=0.2 dB")
#print(f"Apl = {Apl:.4f} dB")
# part-4 *****GOOD*****
print("--------part-5-------------")
print(f"fls = {fls/G:.4f} GHz")
print(f"fhs = {fhs/G:.4f} GHz")
print(f"Als = {Als:.4f} dB")
print(f"Ahs = {Ahs:.4f} dB")
# part-5 *****GOOD*****
print("--------part-6-------------")
print(f"Ap1 = {Ap1:.4f} W/W")
print(f"Xp = {Xp:.4f}")
print(f"E = {E:.4f}")
#print(f"Epsilon= {Epsilon:.4f}")
# part-6 *****GOOD*****
print("--------part-7-------------")
print(f"Als1 = {Als1:.4f} W/W")
print(f"Xls = {Xls:.4f} W/W")
print(f"root_Xls = {root_Xls:.4f}")
# part-7 *****GOOD*****
print("--------part-8-------------")
print(f"Omega_ls = {Omega_ls:.4f}")
print(f"xy = {xy:.4f}")
print(f"y = {y:.4f}")
print(f"Nls = {Nls:.4f}")
# part-8 *****GOOD*****
print("--------part-9-------------")
print(f"Ahs1 = {Ahs1:.4f} W/W")
print(f"Xhs = {Xhs:.4f} W/W")
print(f"root_Xhs = {root_Xhs:.4f}")
print(f"Omega_hs = {Omega_hs:.4f}")
print(f"Ahs1 = {Ahs1:.4f} W/W")
print(f"Nhs = {Nhs:.4f} ")
# part-9 *****GOOD*****
print("--------part-10-------------")
print(f"Nc = {Nc:.4f}")
print(f"Nd = {Nd:.4f}")
# part-10 *****GOOD*****
print("--------part-11-------------")
print("Chebyshev 0.2 dB Ripple") 
print(f"g0 = {g0:.4f}")
print(f"g1 = {g1:.4f}")
print(f"g2 = {g2:.4f}")
print(f"g3 = {g3:.4f}")
print(f"g4 = {g4:.4f}")
# part-11 *****GOOD*****
print("--------part-12-------------")
print(f"Z0 = {Z0:.4f}")
print(f"Y0 = {Y0*1000:.4f} mS")
print(f"w0 = {w0*n:.4f} Grad/S")
# part-12 *****GOOD*****
print("--------part-13-------------")
print("Components")
print(f"C1 = {C1*1e12:.4f} pF")
print(f"L1 = {L1*1e12:.4f} pH")
print(f"L2 = {L2*1e9:.4f} nH")
print(f"C2 = {C2*1e15:.4f} pF")
print(f"C3 = {C3*1e12:.4f} pF")
print(f"L3 = {L3*1e12:.4f} pH")
# part-13 *****GOOD*****
print("--------part-17-------------")
print("Coupled Line Values")
print("Section 01")
print(f"j01 = {j01*k:.4f} mS/S")
print(f"Z0e_01 = {Z0e_01:.4f} Ohms")
print(f"Z0o_01 = {Z0o_01:.4f} Ohms")
print("Section 12")
print(f"j12 = {j12*k:.4f} mS/S")
print(f"Z0e_12 = {Z0e_12:.4f} Ohms")
print(f"Z0o_12 = {Z0o_12:.4f} Ohms")
print("Section 23")
print(f"j23 = {j23*k:.4f} mS/S")
print(f"Z0e_23 = {Z0e_23:.4f} Ohms")
print(f"Z0o_23 = {Z0o_23:.4f} Ohms")
print("Section 34")
print(f"j34 = {j34*k:.4f} mS/S")
print(f"Z0e_34 = {Z0e_34:.4f} Ohms")
print(f"Z0o_34 = {Z0o_34:.4f} Ohms")
# part-17 *****GOOD*****
