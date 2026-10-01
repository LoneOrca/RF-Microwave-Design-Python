import numpy as np
import scipy
import skrf as rf
import matplotlib.pyplot as plt
from ABCD_Shunt_OC_Stub import ABCD_Shunt_OC_Stub
from ABCD_Shunt_SC_Stub import ABCD_Shunt_SC_Stub
from ABCD_to_S import ABCD_to_S
from ABCD_TRL import ABCD_TRL
from S_Param_Interp import S_Param_Interp
from S_to_ABCD import S_to_ABCD

#--------Srinivas Nittala--------
# *Maximum Gain Amplifier using MGF4941AL HEMT @ 15GHz*

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
#---------------------------------
Z0 = 50
ZT = 50
f_min = 14*G
f0 = 15*G
f_max = 16*G
df = 5*Meg;
theta_p = 90
#---------------------------------
network = rf.Network("MGF4941AL.s2p")

freq = network.f
S = network.s

#---------------------------------

mag_S11 = 0.435
ang_S11 = 113.6

mag_S12 = 0.121
ang_S12 = -14.6

mag_S21 = 3.514
ang_S21 = -21.3

mag_S22 = 0.152
ang_S22 = -157.0

#"The 6GHz S-parameters were extracted from the official ATF-33143 low noise PHEMT data sheet."
S11 = (mag_S11*np.cos(np.deg2rad(ang_S11)))+1j*(mag_S11*np.sin(np.deg2rad(ang_S11)))
S12 = (mag_S12*np.cos(np.deg2rad(ang_S12)))+1j*(mag_S12*np.sin(np.deg2rad(ang_S12)))
S21 = (mag_S21*np.cos(np.deg2rad(ang_S21)))+1j*(mag_S21*np.sin(np.deg2rad(ang_S21)))
S22 = (mag_S22*np.cos(np.deg2rad(ang_S22)))+1j*(mag_S22*np.sin(np.deg2rad(ang_S22)))

#---Stability Analysis---
dl = (S11*S22) - (S12*S21)
mag_dl = np.abs(dl)
ang_dl = np.angle(dl, deg=True)
K = (1-(mag_S11)**2-(mag_S22)**2+(mag_dl)**2)/(2*(mag_S12*mag_S21))

#---Key Calculations---
B1 = 1+(mag_S11)**2-(mag_S22)**2-(mag_dl)**2
C1 = S11 - (dl)*np.conj(S22)
mag_C1 = np.abs(C1)
ang_C1 = np.angle(C1, deg = True)
B2 = 1+(mag_S22)**2-(mag_S11)**2-(mag_dl)**2
C2 = S22 - (dl)*np.conj(S11)
mag_C2 = np.abs(C2)
ang_C2 = np.angle(C2, deg = True)
#---Reflection Coeffcients---
#---Source Reflection Coeffcient---
Gamma_S1 = (B1 + (np.sqrt((B1)**2-(4*(mag_C1)**2))))/(2*C1)
mag_Gamma_S1 = np.abs(Gamma_S1)
ang_Gamma_S1 = np.angle(Gamma_S1, deg = True)

Gamma_S2 = (B1 - (np.sqrt((B1)**2-(4*(mag_C1)**2))))/(2*C1)
mag_Gamma_S2 = np.abs(Gamma_S2)
ang_Gamma_S2 = np.angle(Gamma_S2, deg = True)
if abs(Gamma_S1) < 1:
    Gamma_S = Gamma_S1
else:
    Gamma_S = Gamma_S2
#---Load Reflection Coeffcient---
Gamma_L1 = (B2 + (np.sqrt((B2)**2-(4*(mag_C2)**2))))/(2*C2)
mag_Gamma_L1 = np.abs(Gamma_L1)
ang_Gamma_L1 = np.angle(Gamma_L1, deg = True)

Gamma_L2 = (B2 - (np.sqrt((B2)**2-(4*(mag_C2)**2))))/(2*C2)
mag_Gamma_L2 = np.abs(Gamma_L2)
ang_Gamma_L2 = np.angle(Gamma_L2, deg = True)
if abs(Gamma_L1) < 1:
    Gamma_L = Gamma_L1
else:
    Gamma_L = Gamma_L2
#---Input Reflection Coeffcient---
Gamma_IN = (S11)+((S12*S21*Gamma_L)/(1-S22*Gamma_L))
mag_Gamma_IN = np.abs(Gamma_IN)
ang_Gamma_IN = np.angle(Gamma_IN, deg = True)
#---Output Reflection Coeffcient---
Gamma_OUT = (S22)+((S12*S21*Gamma_S)/(1-S11*Gamma_S))
mag_Gamma_OUT = np.abs(Gamma_OUT)
ang_Gamma_OUT = np.angle(Gamma_OUT, deg = True)
#---Gain Calculation---
MAG = ((mag_S21/mag_S12)*(K-(np.sqrt(K**2-1))))
MAG_dB = 10*np.log10(MAG)
X = ((1-(mag_Gamma_S2)**2)*((mag_S21)**2)*(1-(mag_Gamma_L2)**2))
A = 1 - S11*Gamma_S
B = 1 - S22*Gamma_L
C = S21*S12*Gamma_S*Gamma_L
Y = (A*B - C)**2
Z = np.abs(Y)
Gt = X/Z
Gt_dB = 10*np.log10(Gt)
#---Matching Networks---
#---IMN---
Gamma_IMN = np.abs(Gamma_IN-np.conj(Gamma_S))/np.abs(1-(Gamma_IN)*(Gamma_S))
VSWR_IMN = (1+Gamma_IMN)/(1-Gamma_IMN)
ML_IMN  = 1-(np.abs(Gamma_IMN))**2
ML_IMN_dB = 10*np.log10(ML_IMN)
#---OMN---
Gamma_OMN = np.abs(Gamma_OUT-np.conj(Gamma_L))/np.abs(1-(Gamma_OUT)*(Gamma_L))
VSWR_OMN = (1+Gamma_OMN)/(1-Gamma_OMN)
ML_OMN  = 1-(np.abs(Gamma_OMN))**2
ML_OMN_dB = 10*np.log10(ML_OMN)
#---IMN DESIGN---
dI2_lambda = 0.3464
theta_I2 = 360*dI2_lambda
theta_I2_2 = theta_I2*2
dI1_lambda = 0.0741
theta_I1 = 360*dI1_lambda
theta_I1_2 = theta_I1*2
#---OMN DESIGN---
dO2_lambda = 0.1138
theta_O2 = 360*dO2_lambda
theta_O2_2 = theta_O2*2
dO1_lambda = 0.1545
theta_O1 = 360* dO1_lambda
theta_O1_2 = theta_O1*2
#---NETWORK CASCADING---
#---IMN---TRANSISTOR---OMN---
freq_Amp = np.arange(f_min, f_max + df, df) #creates the frequency array.
I_f0_Amp = np.isin(freq_Amp, f0) #checks where f0 exists in the frequency array.
N_Freq = len(freq_Amp) #gives the number of frequency points
S_Amp = np.zeros((N_Freq, 2, 2), dtype=complex) #creates the empty 2×2 S-parameter matrix for every frequency
for kk in range(N_Freq):
 fk = freq_Amp[kk]
 theta_P_fk = (fk/f0)*theta_p
 theta_I1_fk = (fk/f0)*theta_I1
 theta_I2_fk = (fk/f0)*theta_I2
 theta_O1_fx = (fk/f0)*theta_O1
 theta_O2_fx = (fk/f0)*theta_O2
 T0 =  ABCD_TRL(Z0, theta_P_fk)
 TI1 = ABCD_Shunt_SC_Stub(Z0, theta_I1_fk)
 TI2 = ABCD_TRL(Z0, theta_I2_fk)
 TO2 = ABCD_TRL(Z0, theta_O2_fx)
 TO1 = ABCD_Shunt_OC_Stub(Z0, theta_O1_fx)
 Sx = S_Param_Interp(S, freq, fk)
 Ts = S_to_ABCD(Sx, Z0)
 T = T0 @ T0 @ TI1 @ TI2 @ Ts @ TO2 @ TO1 @ T0 @ T0
 S_Amp[kk, :, :] = ABCD_to_S(T, [Z0, Z0])
#-----Parameters Extraction-----------------
S11_Amp = S_Amp[:, 0, 0]
S11_Amp_Mag = np.abs(S11_Amp)
S11_Amp_dB = 20 * np.log10(S11_Amp_Mag)

S21_Amp = S_Amp[:, 1, 0]
S21_Amp_Mag = np.abs(S21_Amp)
S21_Amp_dB = 20 * np.log10(S21_Amp_Mag)

S22_Amp = S_Amp[:, 1, 1]
S22_Amp_Mag = np.abs(S22_Amp)
S22_Amp_dB = 20 * np.log10(S22_Amp_Mag)

S12_Amp = S_Amp[:, 0, 1]
S12_Amp_Mag = np.abs(S12_Amp)
S12_Amp_dB = 20 * np.log10(S12_Amp_Mag)
#------ S-parameters at f0------
S_Amp_f0 = S_Amp[I_f0_Amp, :, :]

S11_Amp_f0 = S11_Amp[I_f0_Amp]
S11_Amp_f0_dB = S11_Amp_dB[I_f0_Amp]

S21_Amp_f0 = S21_Amp[I_f0_Amp]
S21_Amp_f0_dB = S21_Amp_dB[I_f0_Amp]

S22_Amp_f0 = S22_Amp[I_f0_Amp]
S22_Amp_f0_dB = S22_Amp_dB[I_f0_Amp]

S12_Amp_f0 = S12_Amp[I_f0_Amp]
S12_Amp_f0_dB = S12_Amp_dB[I_f0_Amp]

S21_Amp_f0_Mag = abs(S21_Amp_f0)
#---------------------------------
#		|
#		|
#		|			
# Amplifier values for S11,S12,S21,S22nad Gt @ 15GHZ
#		|
#		|
#		|
	# In progress
#-------------Graphs---------------
plt.figure()

plt.plot(freq_Amp / G, S21_Amp_dB, linewidth=6)

plt.scatter(f0 / G, MAG_dB, s=100)
plt.annotate(f"{MAG_dB:.4f} dB", (f0 / G, MAG_dB))

plt.xlabel("Frequency (GHz)")
plt.ylabel("S21 (dB)")
plt.title("S21 Maximum gain Amplifier @ 15GHz")

plt.xlim(14, 16)
plt.ylim(9.5, 13.5)

plt.grid(True)
plt.show()
#------------------
plt.figure()

plt.plot(freq_Amp / G, S11_Amp_dB, 'g',  linewidth = 6)

plt.xlabel("Frequency (GHz)")
plt.ylabel("S11 (dB)")
plt.title("S11 Maximum gain Amplifier @15GHz")

plt.xlim(14, 16)
plt.ylim(-40, 0)

plt.grid(True)
plt.show()
#------------------
plt.figure()

plt.plot(freq_Amp / G, S22_Amp_dB, 'r',  linewidth = 6)

plt.xlabel("Frequency (GHz)")
plt.ylabel("S22 (dB)")
plt.title("S22 Maximum gain Amplifier @15GHz")

plt.xlim(14, 16)
plt.ylim(-40, 0)

plt.grid(True)
plt.show()
#------------------
plt.figure()

plt.plot(freq_Amp / G, S11_Amp_dB, 'r',  linewidth = 6)
plt.plot(freq_Amp / G, S21_Amp_dB, 'b',  linewidth = 6)
plt.plot(freq_Amp / G, S22_Amp_dB, 'g',  linewidth = 6)
plt.plot(freq_Amp / G, S12_Amp_dB, 'c',  linewidth = 6)

plt.xlabel("Frequency (GHz)")
plt.ylabel("Sji (dB)")
plt.title("Sji Maximum gain Amplifier @15GHz")

plt.xlim(14, 16)
plt.ylim(-40, 15)

plt.grid(True)
plt.show()

print("Device: MGF4941AL HEMT")
print("Center Frequency: 15GHz")
print("-----Part-1-----")
print(f"S11= {S11:.4f}")
print(f"S12= {S12:.4f}")
print(f"S21= {S21:.4f}")
print(f"S22= {S22:.4f}")
#---Stability Analysis---
print("-----Part-2-----")
print(f"dl= {dl:.4f}")
print(f"mag_dl= {mag_dl:.4f}")
print(f"ang_dl= {ang_dl:.4f}")
print(f"K= {K:.4f}")
if mag_dl<1 and K>1:
 print("Stability:Unconditionally Stable")
else:
 print("Stability:Conditionally Stable")
#---Key Calculations---
print("-----Part-3-----")
print(f"B1= {B1:.4f}")
print(f"C1= {C1:.4f}")
print(f"mag_C1= {mag_C1:.4f}")
print(f"ang_C1= {ang_C1:.4f}")
print(f"B2= {B2:.4f}")
print(f"C2= {C2:.4f}")
print(f"mag_C2= {mag_C2:.4f}")
print(f"ang_C2= {ang_C2:.4f}")
#---Reflection Coeffcients---
print("-----Part-4-----")
print("-----Reflection Coeffcients-----")
print("-----Source Reflection Coeffcient-----")
print(f"Gamma_S1= {Gamma_S1:.4f}")
print(f"mag_Gamma_S1= {mag_Gamma_S1:.4f}")
print(f"ang_Gamma_S1= {ang_Gamma_S1:.4f}")
print("-----------------")
print(f"Gamma_S2= {Gamma_S2:.4f}")
print(f"mag_Gamma_S2= {mag_Gamma_S2:.4f}")
print(f"ang_Gamma_S2= {ang_Gamma_S2:.4f}")
print(f"Gamma_S = {np.abs(Gamma_S):.4f} ∠ {np.angle(Gamma_S, deg = True):.4f}°")

print("-----Load Reflection Coeffcient-----")
print(f"Gamma_L1= {Gamma_L1:.4f}")
print(f"mag_Gamma_L1= {mag_Gamma_L1:.4f}")
print(f"ang_Gamma_L1= {ang_Gamma_L1:.4f}")
print("-----------------")
print(f"Gamma_L2= {Gamma_L2:.4f}")
print(f"mag_Gamma_L2= {mag_Gamma_L2:.4f}")
print(f"ang_Gamma_L2= {ang_Gamma_L2:.4f}")
print(f"Gamma_L = {np.abs(Gamma_L):.4f} ∠ {np.angle(Gamma_L, deg = True):.4f}°")
print("-----Input Reflection Coeffcient-----")
print(f"Gamma_IN= {Gamma_IN:.4f}")
print(f"mag_Gamma_IN= {mag_Gamma_IN:.4f}")
print(f"ang_Gamma_IN= {ang_Gamma_IN:.4f}")
print(f"Gamma_IN_Polar = {np.abs(Gamma_IN):.4f} ∠ {np.angle(Gamma_IN, deg = True):.4f}°")
print("-----Output Reflection Coeffcient-----")
print(f"Gamma_OUT= {Gamma_OUT:.4f}")
print(f"mag_Gamma_OUT= {mag_Gamma_OUT:.4f}")
print(f"ang_Gamma_OUT= {ang_Gamma_OUT:.4f}")
print(f"Gamma_OUT_Polar = {np.abs(Gamma_OUT):.4f} ∠ {np.angle(Gamma_OUT, deg = True):.4f}°")
if np.isclose(Gamma_IN, np.conj(Gamma_S)) and np.isclose(Gamma_OUT, np.conj(Gamma_L)):
    print("Matching Condition: Maximum gain achieved.")
else:
    print("Matching Condition: Maximum gain not achieved.")
print("-----Part-5-----")
print("-----Gain Calculation-----")
print(f"MAG= {MAG:.4f} W/W")
print(f"MAG_dB= {MAG_dB:.4f} dB")
print(f"X= {X:.4f}")
print(f"A= {A:.4f}")
print(f"B= {B:.4f}")
print(f"C= {C:.4f}")
print(f"Y= {Y:.4f}")
print(f"Z= {Z:.4f}")
print(f"Gt= {Gt:.4f}W/W")
print(f"Gt_dB= {Gt_dB:.4f}dB")
if np.isclose(MAG_dB, Gt_dB) and np.isclose(MAG, Gt):
    print("Maximum Gain condition verified!")
else:
    print("Check the Matching condition!")
print("-----Part-6-----")
print("-----Matching Networks-----")
print("-----IMN-----")
print(f"Gamma_IMN= {Gamma_IMN:.4f}")
print(f"VSWR_IMN= {VSWR_IMN:.4f}")
print(f"ML_IMN= {ML_IMN:.4f} W/W")
print(f"ML_IMN_dB= {ML_IMN_dB:.4f} dB")
print("-----OMN-----")
print(f"Gamma_OMN= {Gamma_OMN:.4f}")
print(f"VSWR_OMN= {VSWR_OMN:.4f}")
print(f"ML_OMN= {ML_OMN:.4f} W/W")
print(f"ML_OMN_dB= {ML_OMN_dB:.4f} dB")
print("-----IMN Design-----")
print(f"theta_I2= {theta_I2:.4f}°")
print(f"theta_I2_2= {theta_I2_2:.4f}°")
print(f"theta_I1= {theta_I1:.4f}°")
print(f"theta_I1_2= {theta_I1_2:.4f}°")
print("-----OMN Design-----")
print(f"theta_O2= {theta_O2:.4f}°")
print(f"theta_O2_2= {theta_O2_2:.4f}°")
print(f"theta_O1= {theta_O1:.4f}°")
print(f"theta_O1_2= {theta_O1_2:.4f}°")
#print("-----S-parameters at f0-----")
#print(f"S11_Amp_f0= {S11_Amp_f0:.4f}")
#print(f"S21_Amp_f0= {S21_Amp_f0:.4f}")
#print(f"S12_Amp_f0= {S12_Amp_f0:.4f}")
#print(f"S22_Amp_f0= {S22_Amp_f0:.4f}")
