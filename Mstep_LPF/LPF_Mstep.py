import numpy as np
import matplotlib.pyplot as plt
from ABCD_TRL import ABCD_TRL
from ABCD_to_S import ABCD_to_S

#-----------Srinivas Nittala-----------------
# Lumped Element Filter to Stepped-Impedance Transmission Line


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
f_min = 1
f_max = 2.5*G
df = 5*Meg
j = 1j
#-----------Part-1----------------
IL_max = 0.75
S21_IL_max = -(IL_max)
S21_IL_min = 10**((S21_IL_max)/20)
S11_IL_max = np.sqrt(1-(S21_IL_min)**2)
S11_IL = 20*np.log10(S11_IL_max)
#-----------Part-2----------------			
RL_min = 13.4672
S11_RL = -(RL_min)
S11_RL_max = 10**((S11_RL)/20)
S21_RL_min = np.sqrt(1-(S11_RL_max)**2)
S21_RL = 20*np.log10(S21_RL_min)
#-----------Part-3----------------	
fs = 2*G
fp = 1*G
Ap = -(S21_RL)
IL_min = 20;
As = IL_min;
Ap1 = 10**(Ap/10)
Xp = (Ap1)-1
Xp1 = Xp*k
E = np.sqrt(Xp)
As1 = 10**(As/10)
Xs = As1 -1
Xs_root = np.sqrt(Xs)
omega_s = fs/fp
Np = np.arccosh((1/E)*(Xs_root))/np.arccosh(omega_s)
N = int(np.ceil(Np))
if N % 2 == 0:
  N += 1
Nc = N
#-----------Part-4----------------
g0 = 1.0000
g1 = 1.3394
g2 = 1.3370
g3 = 2.1661
g4 = 1.3370
g5 = 1.3394
g6 = 1.0000
#-----------Part-5----------------
Z0 = 50
Y0 = 1/Z0
Y01 = Y0*k
z0L = 20
y0L = 1/z0L
z0H = 120
y0H = 1/z0H
fc = 1*G
Wc = (2*np.pi*fc)
lbda = C/fc
#-----------Part-6----------------
#---------COMPONENTS--------------
C1 = (1/Wc)*Y0*g1
C3 = (1/Wc)*Y0*g3
C5 = (1/Wc)*Y0*g5
L2 = (1/Wc)*Z0*g2
L4 = (1/Wc)*Z0*g4
#-----------Part-7----------------
#--------Electrical Lengths-------
thetak1 = (Y0/y0L)*g1
thetak2 = (Z0/z0H)*g2
thetak3 = (Y0/y0L)*g3
thetak4 = (Z0/z0H)*g4
thetak5 = (Y0/y0L)*g5
#-------- radians to degree conversion--------
thetak1_degrees = np.degrees(thetak1)
thetak2_degrees = np.degrees(thetak2)
thetak3_degrees = np.degrees(thetak3)
thetak4_degrees = np.degrees(thetak4)
thetak5_degrees = np.degrees(thetak5)
#-----------Part-8----------------
freq = np.arange(f_min, f_max + df, df) #Create Frequency Range
N_Freq = len(freq) #Count frequencies
S_Filter = np.zeros((N_Freq, 2, 2), dtype=complex) #Create apce for S param
for kk in range(N_Freq): #Loop through every frequency
 fk = freq[kk] #Pick Current Frequency
 T0 = np.eye(2)
 T1 = ABCD_TRL(z0L,thetak1_degrees*fk/fp)
 T2 = ABCD_TRL(z0H,thetak2_degrees*fk/fp)
 T3 = ABCD_TRL(z0L,thetak3_degrees*fk/fp)
 T4 = ABCD_TRL(z0H,thetak4_degrees*fk/fp)
 T5 = ABCD_TRL(z0L,thetak5_degrees*fk/fp)
 T = T0 @ T1 @ T2 @ T3 @ T4 @ T5
 S_Filter[kk, :, :] = ABCD_to_S(T, [Z0, Z0])
#-----------Part-9----------------
S11_Filter = S_Filter[:, 0, 0]
S11_Filter_Mag = np.abs(S11_Filter)
S11_Filter_dB = 20 * np.log10(S11_Filter_Mag)

S21_Filter = S_Filter[:, 1, 0]
S21_Filter_Mag = np.abs(S21_Filter)
S21_Filter_dB = 20 * np.log10(S21_Filter_Mag)
#-----------Part-10----------------
plt.figure()

plt.plot(freq / G, S11_Filter_dB, 'g',  linewidth = 6)

plt.xlabel("Frequency (GHz)")
plt.ylabel("S11 (dB)")
plt.title("S11 Stepped Impedence Low-Pass Filter")

plt.xlim(0, 2.5)
plt.ylim(-40, 0)

plt.grid(True)
plt.show()
#------------
plt.figure()

plt.plot(freq / G, S21_Filter_dB, 'r', linewidth = 6)

plt.xlabel("Frequency (GHz)")
plt.ylabel("S21 (dB)")
plt.title("S21 Stepped Impedence Low-Pass Filter")

plt.xlim(0, 1.2)
plt.ylim(-1, 0)

plt.grid(True)
plt.show()
#-------------
plt.figure()

plt.plot(freq / G, S11_Filter_dB, 'g',  linewidth = 3)
plt.plot(freq / G, S21_Filter_dB, 'r', linewidth = 3)
plt.xlabel("Frequency (GHz)")
plt.ylabel("S-Parameters (dB)")
plt.title("S11 and S21 of Stepped-Impedance LPF")

plt.xlim(0, 2.5)
plt.ylim(-50, 0)

plt.grid(True)
plt.show()
#---------------
print("--------part-1-------------")
print(f"IL_max = {IL_max:.4f}")
print(f"S21_IL_max = {S21_IL_max:.4f}")
print(f"S21_IL_min = {S21_IL_min:.4f}")
print(f"S11_IL_max = {S11_IL_max:.4f} W/W")
print(f"S11_IL = {S11_IL:.4f} dB")
# Part-1 *****GOOD*****
print("---------part-2-------------")
print(f"RL_min = {RL_min:.4f}")
print(f"S11_RL = {S11_RL:.4f}")
print(f"S11_RL_max = {S11_RL_max:.4f}")
print(f"S21_RL_min = {S21_RL_min:.4f} W/W")
print(f"S21_RL = {S21_RL:.4f} dB")
# Part-2 *****GOOD*****
print("----------part-3------------")
print(f"Ap = {Ap:.4f} dB")
print(f"As = {As:.4f} dB")
print(f"Ap1 = {Ap1:.4f} W/W")
print(f"Xp = {Xp:.4f}")
print(f"Xp1 = {Xp1:.4f} mW/W")
print(f"E = {E:.4f}")
print(f"As1 = {As1:.4f}")
print(f"Xs = {Xs:.4f}")
print(f"Xs_root = {Xs_root:.4f}")
print(f"omega_s = {omega_s:.4f} W/W")
print(f"Np = {Np:.4f} ")
print(f"Filter Order = {Nc:.4f} ")
# Part-3 *****GOOD*****
print("----------part-4------------")
print("Chebyshev 0.2 dB Ripple") 
print(f"g0 = {g0:.4f} ")
print(f"g1 = {g1:.4f} ")
print(f"g2 = {g2:.4f} ")
print(f"g3 = {g3:.4f} ")
print(f"g4 = {g4:.4f} ")
print(f"g5 = {g5:.4f} ")
print(f"g6 = {g6:.4f} ")
# Part-4 *****GOOD*****
print("----------part-5------------")
print(f"Z0 = {Z0:.4f} Ohms")
print(f"Y0 = {Y0*k:.4f} S")
print(f"z0L = {z0L:.4f} Ohms")
print(f"y0L = {y0L*k:.4f} S")
print(f"z0H = {z0H:.4f} Ohms")
print(f"y0H = {y0H*k:.4f} S")
# Part-5 *****GOOD*****
print("----------part-6------------")
print(f"C1 = {C1/p:.4f} pF")
print(f"L2 = {L2/n:.4f} nH")
print(f"C3 = {C3/p:.4f} pF")
print(f"L4 = {L4/n:.4f} nH")
print(f"C5 = {C5/p:.4f} pF")
# Part-6 *****GOOD*****
print("----------part-7------------")
print("----------Electrical Lengths------------")
print(f"thetak1 = {thetak1*1e2:.4f} rad")
print(f"thetak2 = {thetak2*1e2:.4f} rad")
print(f"thetak3 = {thetak3*1e2:.4f} rad")
print(f"thetak4 = {thetak4*1e2:.4f} rad")
print(f"thetak5 = {thetak5*1e2:.4f} rad")
# Part-7 *****GOOD*****
print("----------part-8------------")
print("----------rad to deg------------")
print(f"thetak1_degrees = {thetak1_degrees:.4f} degrees")
print(f"thetak2_degrees = {thetak2_degrees:.4f} degrees")
print(f"thetak3_degrees = {thetak3_degrees:.4f} degrees")
print(f"thetak4_degrees = {thetak4_degrees:.4f} degrees")
print(f"thetak5_degrees = {thetak5_degrees:.4f} degrees")
