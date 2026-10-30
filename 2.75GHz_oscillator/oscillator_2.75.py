import numpy as np
import scipy
#import skrf as rf
import matplotlib.pyplot as plt
#------Srinivas Nittala------
# *Oscillator using BJT @ 2.75GHz*

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
#---------------------
Z0 = 50
Y0 = 1/Z0
f0 = 2.75*G
#----S-parameters---------
mag_S11 = 0.6162
ang_S11 = 169.8792

mag_S12 = 0.1487
ang_S12 = 66.5075

mag_S21 = 1.5824
ang_S21 = 53.6005

mag_S22 = 0.3407
ang_S22 = -45.1474

#----S-parameters (Polar to rect)----
S11 = (mag_S11*np.cos(np.deg2rad(ang_S11)))+1j*(mag_S11*np.sin(np.deg2rad(ang_S11)))
S12 = (mag_S12*np.cos(np.deg2rad(ang_S12)))+1j*(mag_S12*np.sin(np.deg2rad(ang_S12)))
S21 = (mag_S21*np.cos(np.deg2rad(ang_S21)))+1j*(mag_S21*np.sin(np.deg2rad(ang_S21)))
S22 = (mag_S22*np.cos(np.deg2rad(ang_S22)))+1j*(mag_S22*np.sin(np.deg2rad(ang_S22)))
SCE = np.array([
    [S11, S12],
    [S21, S22]
])

#---Stability Analysis---
dl = (S11*S22) - (S12*S21)
mag_dl = np.abs(dl)
ang_dl = np.angle(dl, deg=True)
K = (1-(mag_S11)**2-(mag_S22)**2+(mag_dl)**2)/(2*(mag_S12*mag_S21))

#----S-parameters to Y-parameters----
# YCE = Y0 ({[U]-[SCE]}{U]+[SCE]}**-1)
U = np.eye(2)
A_S2Y = U - SCE
B_S2Y = U + SCE
B_S2Y_Inv = np.linalg.inv(B_S2Y)
YCE_S2Y = A_S2Y @ B_S2Y_Inv
YCE = Y0*(YCE_S2Y)
YCE_full_precision = YCE
#------------------------
YCE_ybb = YCE_full_precision[0, 0]
YCE_ybc = YCE_full_precision[0, 1]
YCE_ycb = YCE_full_precision[1, 0]
YCE_ycc = YCE_full_precision[1, 1]
#-----Conversion of 2x2 to 3x3 matrix[YCE_full_precision to Y_I]------------
YCE_ybe = - (YCE_ybb + YCE_ybc)
YCE_yce = - (YCE_ycb + YCE_ycc)
YCE_yeb = - (YCE_ybb + YCE_ycb)
YCE_yec = - (YCE_ybc + YCE_ycc)
YCE_yee = - (YCE_ybe + YCE_yce)
YCE_yee_2 = - (YCE_yeb + YCE_yec)
#-----Rectangular to Polar ------------
mag_YCE_ybb = np.abs(YCE_ybb)
ang_YCE_ybb = np.angle(YCE_ybb, deg = True)

mag_YCE_ybc = np.abs(YCE_ybc)
ang_YCE_ybc = np.angle(YCE_ybc, deg = True)

mag_YCE_ybe = np.abs(YCE_ybe)
ang_YCE_ybe = np.angle(YCE_ybe, deg = True)

mag_YCE_ycb = np.abs(YCE_ycb)
ang_YCE_ycb = np.angle(YCE_ycb, deg = True)

mag_YCE_ycc = np.abs(YCE_ycc)
ang_YCE_ycc = np.angle(YCE_ycc, deg = True)

mag_YCE_yce = np.abs(YCE_yce)
ang_YCE_yce = np.angle(YCE_yce, deg = True)

mag_YCE_yeb = np.abs(YCE_yeb)
ang_YCE_yeb = np.angle(YCE_yeb, deg = True)

mag_YCE_yec = np.abs(YCE_yec)
ang_YCE_yec = np.angle(YCE_yec, deg = True)

mag_YCE_yee = np.abs(YCE_yee)
ang_YCE_yee = np.angle(YCE_yee, deg = True)

mag_YCE_yee_2 = np.abs(YCE_yee_2)
ang_YCE_yee_2 = np.angle(YCE_yee_2, deg = True)

YI = np.array([
    [YCE_ybb, YCE_ybc, YCE_ybe ],
    [YCE_ycb, YCE_ycc, YCE_yce ],
    [YCE_yeb, YCE_yec, YCE_yee ]
])
# Let's reduce the 3x3 matrix to 2x2 {[YI] to [YCB]}

YCB = np.array([
    [YCE_yee, YCE_yec ],
    [YCE_yce, YCE_ycc ]
])
#----Y-parameters to S-parameters----
# SCB = ({[U]Y0-[YCB]}{[U]Y0+[YCB]}**-1)
A_YCB = U*Y0
B_YCB = A_YCB - YCB
C_YCB = A_YCB + YCB
D_YCB = np.linalg.inv(C_YCB)
SCB = B_YCB @ D_YCB
#------------------------
SCB_S11 = SCB[0, 0]
SCB_S12 = SCB[0, 1]
SCB_S21 = SCB[1, 0]
SCB_S22 = SCB[1, 1]
#-----Rectangular to Polar ------------
mag_SCB_S11 = np.abs(SCB_S11)
ang_SCB_S11 = np.angle(SCB_S11, deg = True)

mag_SCB_S12 = np.abs(SCB_S12)
ang_SCB_S12 = np.angle(SCB_S12, deg = True)

mag_SCB_S21 = np.abs(SCB_S21)
ang_SCB_S21 = np.angle(SCB_S21, deg = True)

mag_SCB_S22 = np.abs(SCB_S22)
ang_SCB_S22 = np.angle(SCB_S22, deg = True)
#---SCB-Stability Analysis---
dl_SCB = (SCB_S11*SCB_S22) - (SCB_S12*SCB_S21)
mag_dl_SCB = np.abs(dl_SCB)
ang_dl_SCB = np.angle(dl_SCB, deg = True)
K_SCB = (1-(mag_SCB_S11)**2-(mag_SCB_S22)**2+(mag_dl_SCB)**2)/(2*(mag_SCB_S12*mag_SCB_S21))
#----S-parameters to Z-parameters----
# ZCB = (Z0{[U]+[SCB]}{[U]-[SCB]}**-1)
A_ZCB = U + SCB
B_ZCB = U - SCB
C_ZCB = np.linalg.inv(B_ZCB)
D_ZCB = A_ZCB @ C_ZCB
ZCB_precison = Z0*D_ZCB
ZCB = ZCB_precison
#------------------------
ZCB_S11 = ZCB[0, 0]
ZCB_S12 = ZCB[0, 1]
ZCB_S21 = ZCB[1, 0]
ZCB_S22 = ZCB[1, 1]
#-----Rectangular to Polar ------------
mag_ZCB_S11 = np.abs(ZCB_S11)
ang_ZCB_S11 = np.angle(ZCB_S11, deg = True)

mag_ZCB_S12 = np.abs(ZCB_S12)
ang_ZCB_S12 = np.angle(ZCB_S12, deg = True)

mag_ZCB_S21 = np.abs(ZCB_S21)
ang_ZCB_S21 = np.angle(ZCB_S21, deg = True)

mag_ZCB_S22 = np.abs(ZCB_S22)
ang_ZCB_S22 = np.angle(ZCB_S22, deg = True)

#-Add Base Feedback Reactance (X_B)-
X_B = np.linspace(-100, 100, 1000) 
# we can keep 9 or 1000, if we keep 1000 then graph will be smooth with no spikes.
FB = np.array([
    [1, 1],
    [1, 1]
])
ZCB_FB = np.zeros((len(X_B), 2, 2), dtype=complex)
for i, XB in enumerate(X_B):
   ZCB_FB[i] = ZCB + 1j * XB * FB
S_CB_FB = np.zeros((len(X_B), 2, 2), dtype=complex)
for i in range(len(X_B)):
    Z = ZCB_FB[i]
    S_CB_FB[i] = (Z - Z0 * U) @ np.linalg.inv(Z + Z0 * U)
#----magnitude of the S-parameters----
mag_S11_S_CB_FB = np.abs(S_CB_FB[:, 0, 0])
mag_S12_S_CB_FB = np.abs(S_CB_FB[:, 0, 1])
mag_S21_S_CB_FB = np.abs(S_CB_FB[:, 1, 0])
mag_S22_S_CB_FB = np.abs(S_CB_FB[:, 1, 1])
#----Stability calculations----
#----Delta Values----
delta = (
    S_CB_FB[:, 0, 0] * S_CB_FB[:, 1, 1]
    - S_CB_FB[:, 0, 1] * S_CB_FB[:, 1, 0]
)
mag_delta_S_CB_FB = np.abs(delta)
#-----Stability Factor K-----

K_S_CB_FB = (
    1
    - mag_S11_S_CB_FB**2
    - mag_S22_S_CB_FB**2
    + mag_delta_S_CB_FB**2
) / (
    2 * mag_S12_S_CB_FB * mag_S21_S_CB_FB
)

#-----Mu Stability Factors-----

S11_FB = S_CB_FB[:, 0, 0]
S12_FB = S_CB_FB[:, 0, 1]
S21_FB = S_CB_FB[:, 1, 0]
S22_FB = S_CB_FB[:, 1, 1]

mu_out = (
    1 - mag_S11_S_CB_FB**2
) / (
    np.abs(S22_FB - delta * np.conj(S11_FB))
    + np.abs(S12_FB * S21_FB)
)

mu_in = (
    1 - mag_S22_S_CB_FB**2
) / (
    np.abs(S11_FB - delta * np.conj(S22_FB))
    + np.abs(S12_FB * S21_FB)
)
#-----Plot K,Mu_in and Mu_out versus Feedback Reactance-----
plt.figure()

plt.plot(X_B, K_S_CB_FB, 'g', linewidth=3, label='K')
plt.plot(X_B, mu_in, 'r', linewidth=3, label='Mu_in')
plt.plot(X_B, mu_out, 'b', linewidth=3, label='Mu_out')

plt.xlabel("Feedback Reactance (Ohm)")
plt.ylabel("Stability Factors")
plt.title("Stability Factors versus Feedback Reactance")

plt.xlim(-100, 100)

plt.grid(True)
plt.legend()
plt.show()
#-----Plot |S11| and |S22| versus Feedback Reactance-----
plt.figure()

plt.plot(X_B, mag_S11_S_CB_FB, 'r', linewidth=3, label='|S11|')
plt.plot(X_B, mag_S22_S_CB_FB, 'b', linewidth=3, label='|S22|')

plt.xlabel("Feedback Reactance (Ohm)")
plt.ylabel("Magnitude")
plt.title("|S11| and |S22| versus Feedback Reactance")
# That sharp dip or spike you are seeing right @ S11 is the X_B_1
plt.xlim(-100, 100)
plt.ylim(0, 2)

plt.grid(True)
plt.legend()
plt.show()

#- Rearrange the Feedback Reactance (X_B_1 = 25 ohms)-
X_B_1 = 25.0542
LB = (X_B_1)/(2*np.pi*(2.75*G))
ZCB_FB_1 = ZCB + 1j * X_B_1 * FB
#------------------------
ZCB_FB_1_S11 = ZCB_FB_1[0, 0]
ZCB_FB_1_S12 = ZCB_FB_1[0, 1]
ZCB_FB_1_S21 = ZCB_FB_1[1, 0]
ZCB_FB_1_S22 = ZCB_FB_1[1, 1]
#-----Rectangular to Polar ------------
mag_ZCB_FB_1_S11 = np.abs(ZCB_FB_1_S11)
ang_ZCB_FB_1_S11 = np.angle(ZCB_FB_1_S11, deg = True)

mag_ZCB_FB_1_S12 = np.abs(ZCB_FB_1_S12)
ang_ZCB_FB_1_S12 = np.angle(ZCB_FB_1_S12, deg = True)

mag_ZCB_FB_1_S21 = np.abs(ZCB_FB_1_S21)
ang_ZCB_FB_1_S21 = np.angle(ZCB_FB_1_S21, deg = True)

mag_ZCB_FB_1_S22 = np.abs(ZCB_FB_1_S22)
ang_ZCB_FB_1_S22 = np.angle(ZCB_FB_1_S22, deg = True)
#----Z-parameters to S-parameters----
# SCB_FB_1 = ({Z0[U]+[ZCB_FB_1]}**-1{Z0[U]-[ZCB_FB_1]})
A_SCB_FB_1 = ZCB_FB_1 + Z0*U
B_SCB_FB_1 = np.linalg.inv(A_SCB_FB_1)
C_SCB_FB_1 = ZCB_FB_1 - Z0*U
SCB_FB_1_precison = B_SCB_FB_1 @ C_SCB_FB_1
SCB_FB_1 = SCB_FB_1_precison
#------------------------
SCB_FB_1_S11 = SCB_FB_1[0, 0]
SCB_FB_1_S12 = SCB_FB_1[0, 1]
SCB_FB_1_S21 = SCB_FB_1[1, 0]
SCB_FB_1_S22 = SCB_FB_1[1, 1]
#-----Rectangular to Polar ------------
mag_SCB_FB_1_S11 = np.abs(SCB_FB_1_S11)
ang_SCB_FB_1_S11 = np.angle(SCB_FB_1_S11, deg = True)

mag_SCB_FB_1_S12 = np.abs(SCB_FB_1_S12)
ang_SCB_FB_1_S12 = np.angle(SCB_FB_1_S12, deg = True)

mag_SCB_FB_1_S21 = np.abs(SCB_FB_1_S21)
ang_SCB_FB_1_S21 = np.angle(SCB_FB_1_S21, deg = True)

mag_SCB_FB_1_S22 = np.abs(SCB_FB_1_S22)
ang_SCB_FB_1_S22 = np.angle(SCB_FB_1_S22, deg = True)
#----Stability calculations----
#----Delta Values----
dl_SCB_FB_1 = (SCB_FB_1_S11*SCB_FB_1_S22)-(SCB_FB_1_S12*SCB_FB_1_S21)
mag_dl_SCB_FB_1 = np.abs(dl_SCB_FB_1)
ang_dl_SCB_FB_1 = np.angle(dl_SCB_FB_1, deg = True)
K_SCB_FB_1 = (1-(mag_SCB_FB_1_S11)**2-(mag_SCB_FB_1_S22)**2+(mag_dl_SCB_FB_1)**2)/(2*(mag_SCB_FB_1_S12*mag_SCB_FB_1_S21))

mu_out_SCB_FB_1 = (
1 - mag_SCB_FB_1_S11**2
) / (
    np.abs(SCB_FB_1_S22 - dl_SCB_FB_1 * np.conj(SCB_FB_1_S11))
    + np.abs(SCB_FB_1_S12 * SCB_FB_1_S21)
)

mu_in_SCB_FB_1 = (
    1 - mag_SCB_FB_1_S22**2
) / (
    np.abs(SCB_FB_1_S11 - dl_SCB_FB_1 * np.conj(SCB_FB_1_S22))
    + np.abs(SCB_FB_1_S12 * SCB_FB_1_S21))
#----Output Stability circles/Terminating Stability circle----    
A_Ct = (SCB_FB_1_S22 - (dl_SCB_FB_1)*np.conj(SCB_FB_1_S11))
B_Ct = np.conj(A_Ct)
Ct = B_Ct/((mag_SCB_FB_1_S22)**2-(mag_dl_SCB_FB_1)**2)  

Rt = np.abs((SCB_FB_1_S12*SCB_FB_1_S21)/((mag_SCB_FB_1_S22)**2-(mag_dl_SCB_FB_1)**2) )
#-----Rectangular to Polar ------------
mag_Ct = np.abs(Ct)
ang_Ct = np.angle(Ct, deg = True)
#------------------------

#------------------------
mag_Gamma_T = 0.5000
ang_Gamma_T = 162.0000
Gamma_T = (mag_Gamma_T*np.cos(np.deg2rad(ang_Gamma_T)))+1j*(mag_Gamma_T*np.sin(np.deg2rad(ang_Gamma_T)))

Gamma_IN = (SCB_FB_1_S11)+((SCB_FB_1_S12*SCB_FB_1_S21*Gamma_T)/(1-SCB_FB_1_S22*Gamma_T))
mag_Gamma_IN = np.abs(Gamma_IN)
ang_Gamma_IN = np.angle(Gamma_IN, deg = True)

Z_IN = ((1+Gamma_IN)/(1-Gamma_IN))*Z0

Rr = -(1/3)*Z_IN.real
Xr = -(Z_IN.imag)
Z_R = Rr + 1j*(Xr)

Gamma_R = (Z_R-Z0)/(Z_R+Z0)
mag_Gamma_R = np.abs(Gamma_R)
ang_Gamma_R = np.angle(Gamma_R, deg = True)
#------------------------
#----Smith Chart----
#------------------------
#----IMN Design-Resonator Network----
dR2_lambda = 0.3708
theta_R2 = 360*dR2_lambda
theta_R2_2 = theta_R2*2
dR1_lambda = 0.0648
theta_R1 = 360*dR1_lambda
theta_R1_2 = theta_R1*2
#----OMN Design-Termination Network----
dT2_lambda = 0.1083
theta_T2 = 360*dT2_lambda
theta_T2_2 = theta_T2*2
dT1_lambda = 0.1364
theta_T1 = 360* dT1_lambda
theta_T1_2 = theta_T1*2
#--------Display values--------
print("Device:Oscillator using BJT")
print("Center Frequency: 2.75GHz")
print("-----Part-1-----")
print(f"S11= {S11:.4f}")
print(f"S12= {S12:.4f}")
print(f"S21= {S21:.4f}")
print(f"S22= {S22:.4f}")
print(f"SCE=")
print(np.round(SCE,4))
#---Stability Analysis---
print("-----Stability Analysis-----")
print(f"dl= {dl:.4f}")
print(f"mag_dl= {mag_dl:.4f}")
print(f"ang_dl= {ang_dl:.4f}")
print(f"K= {K:.4f}")
if mag_dl<1 and K>1:
 print("Stability:Unconditionally Stable")
else:
 print("Stability:Conditionally Stable")
print("--------------------")
print("-----S-parameters to Y-parameters-----")
print("YCE = Y0 ({[U]-[SCE]}{U]+[SCE]}**-1)")
print("--------------------")
print("Identity Matrix")
print(f"U=")
print(np.round(U,4))
print(f"A_S2Y=")
print(np.round(A_S2Y,4))
print(f"B_S2Y=")
print(np.round(B_S2Y,4))
print(f"B_S2Y_Inv=")
print(np.round(B_S2Y_Inv,4))
print(f"YCE_S2Y=")
print(np.round(YCE_S2Y,4))
print(f"YCE=")
print(np.round(YCE,4))
print("YCE full precision:")
print(YCE_full_precision)
print("--------------------")
print("--YI matrix elements--")
print("--------------------")
print(f"YCE_ybb= {YCE_ybb:.4f}")
print(f"YCE_ybc= {YCE_ybc:.4f}")
print(f"YCE_ybe= {YCE_ybe:.4f}")
print(f"YCE_ycb= {YCE_ycb:.4f}")
print(f"YCE_ycc= {YCE_ycc:.4f}")
print(f"YCE_yce= {YCE_yce:.4f}")
print(f"YCE_yeb= {YCE_yeb:.4f}")
print(f"YCE_yec= {YCE_yec:.4f}")
print(f"YCE_yee= {YCE_yee:.4f}")
print("--------------------")
print("--rect to polar--")
print("--------------------")
print(f"mag_YCE_ybb= {mag_YCE_ybb*k:.4f}")
print(f"ang_YCE_ybb= {ang_YCE_ybb:.4f}°")
print("--------------------")
print(f"mag_YCE_ybc= {mag_YCE_ybc*k:.4f}")
print(f"ang_YCE_ybc= {ang_YCE_ybc:.4f}°")
print("--------------------")
print(f"mag_YCE_ybe= {mag_YCE_ybe*k:.4f}")
print(f"ang_YCE_ybe= {ang_YCE_ybe:.4f}°")
print("--------------------")
print(f"mag_YCE_ycb= {mag_YCE_ycb*k:.4f}")
print(f"ang_YCE_ycb= {ang_YCE_ycb:.4f}°")
print("--------------------")
print(f"mag_YCE_ycc= {mag_YCE_ycc*k:.4f}")
print(f"ang_YCE_ycc= {ang_YCE_ycc:.4f}°")
print("--------------------")
print(f"mag_YCE_yce= {mag_YCE_yce*k:.4f}")
print(f"ang_YCE_yce= {ang_YCE_yce:.4f}°")
print("--------------------")
print(f"mag_YCE_yeb= {mag_YCE_yeb*k:.4f}")
print(f"ang_YCE_yeb= {ang_YCE_yeb:.4f}°")
print("--------------------")
print(f"mag_YCE_yec= {mag_YCE_yec*k:.4f}")
print(f"ang_YCE_yec= {ang_YCE_yec:.4f}°")
print("--------------------")
print(f"mag_YCE_yee= {mag_YCE_yee*k:.4f}")
print(f"ang_YCE_yee= {ang_YCE_yee:.4f}°")
print("--------------------")
print(f"mag_YCE_yee_2= {mag_YCE_yee_2:.4f}")
print(f"ang_YCE_yee_2= {ang_YCE_yee_2:.4f}°")
print("--------------------")
print("--YI matrix --")
print("YI:")
print(YI)
print("--------------------")
print("--Let's reduce the 3x3 matrix to 2x2 {[YI] to [YCB]} --")
print("--YCB matrix --")
print("YCB:")
print(YCB)
print("--------------------")
print("-----Y-parameters to S-parameters-----")
print("SCB = ({[U]Y0-[YCB]}{[U]Y0+[YCB]}**-1)")
print("--------------------")
print("A_YCB:")
print(A_YCB)
print("B_YCB:")
print(B_YCB)
print("C_YCB:")
print(C_YCB)
print("D_YCB:")
print(D_YCB)
print("SCB:")
print(SCB)
print("--------------------")
print("--rect to polar--")
print("--------------------")
print(f"mag_SCB_S11= {mag_SCB_S11:.4f}")
print(f"ang_SCB_S11= {ang_SCB_S11:.4f}°")
print("--------------------")
print(f"mag_SCB_S12= {mag_SCB_S12:.4f}")
print(f"ang_SCB_S12= {ang_SCB_S12:.4f}°")
print("--------------------")
print(f"mag_SCB_S21= {mag_SCB_S21:.4f}")
print(f"ang_SCB_S21= {ang_SCB_S21:.4f}°")
print("--------------------")
print(f"mag_SCB_S22= {mag_SCB_S22:.4f}")
print(f"ang_SCB_S22= {ang_SCB_S22:.4f}°")
print("--------------------")
print("-----SCB-Stability Analysis-----")
print(f"dl_SCB= {dl_SCB:.4f}")
print(f"mag_dl_SCB= {mag_dl_SCB:.4f}")
print(f"ang_dl_SCB= {ang_dl_SCB:.4f}")
print(f"K_SCB= {K_SCB:.4f}")
if mag_dl_SCB<1 and K_SCB>1:
 print("Stability:Unconditionally Stable")
else:
 print("Stability: UnStable")
 print("--------------------")
print("-----S-parameters to Z-parameters-----")
print("ZCB = (Z0{[U]+[SCB]}{[U]-[SCB]}**-1)")
print("--------------------")
print("A_ZCB:")
print(A_ZCB)
print("B_ZCB:")
print(B_ZCB)
print("C_ZCB:")
print(C_ZCB)
print("D_ZCB:")
print(D_ZCB)
print("ZCB:")
print(ZCB)
print("--------------------")
print("--rect to polar--")
print("--------------------")
print(f"mag_ZCB_S11= {mag_ZCB_S11:.4f}")
print(f"ang_ZCB_S11= {ang_ZCB_S11:.4f}°")
print("--------------------")
print(f"mag_ZCB_S12= {mag_ZCB_S12:.4f}")
print(f"ang_ZCB_S12= {ang_ZCB_S12:.4f}°")
print("--------------------")
print(f"mag_ZCB_S21= {mag_ZCB_S21:.4f}")
print(f"ang_ZCB_S21= {ang_ZCB_S21:.4f}°")
print("--------------------")
print(f"mag_ZCB_S22= {mag_ZCB_S22:.4f}")
print(f"ang_ZCB_S22= {ang_ZCB_S22:.4f}°")
print("--------------------")
print("-----Display Feedback Impedance Matrices-----")
for i, XB in enumerate(X_B):
    print(f"\nFeedback Reactance = {XB} Ohms")
    print("ZCB_FB =")
    print(ZCB_FB[i])
print("--------------------")
print("-----Display S-Parameter Matrices-----")
for i, XB in enumerate(X_B):
    print(f"\nX_B = {XB:.1f} Ohm")
    print(S_CB_FB[i])
print("--------------------")
print("-----Display S-Parameter Magnitudes-----")
for i, XB in enumerate(X_B):
    print(f"\nX_B = {XB:.1f} Ohm")
    print(f"|S11| = {mag_S11_S_CB_FB[i]:.4f}")
    print(f"|S12| = {mag_S12_S_CB_FB[i]:.4f}")
    print(f"|S21| = {mag_S21_S_CB_FB[i]:.4f}")
    print(f"|S22| = {mag_S22_S_CB_FB[i]:.4f}")

print("-----Display Delta-----")
print("--------------------")
for i, XB in enumerate(X_B):
    print(f"X_B = {XB:.1f} Ohm, Delta = {delta[i]}")
print("--------------------")
print("-----Display Magnitude of Delta-----")
for i, XB in enumerate(X_B):
    print(f"X_B = {XB:.1f} Ohm, |Delta| = {mag_delta_S_CB_FB[i]:.4f}")
print("--------------------")
print("-----Display Stability Factor K-----")

for i, XB in enumerate(X_B):
    print(f"X_B = {XB:.1f} Ohm, K = {K_S_CB_FB[i]:.4f}")
print("--------------------")
print("-----Display Mu Stability Factors-----")
for i, XB in enumerate(X_B):
    print(f"\nX_B = {XB:.1f} Ohm")
    print(f"Mu_in  = {mu_in[i]:.4f}")
    print(f"Mu_out = {mu_out[i]:.4f}")
print("--------------------")
print("-----Rearrange the Feedback Reactance-----")
print("-----That sharp dip or spike you are seeing right @ S11 is the X_B_1-----")
print(f"X_B_1= {X_B_1:.4f} ohms")
print(f"LB= {LB*G:.4f} nH")
print("ZCB_FB_1:")
print(ZCB_FB_1)
print("--------------------")
print("--rect to polar--")
print("--------------------")
print(f"mag_ZCB_FB_1_S11= {mag_ZCB_FB_1_S11:.4f}")
print(f"ang_ZCB_FB_1_S11= {ang_ZCB_FB_1_S11:.4f}°")
print("--------------------")
print(f"mag_ZCB_FB_1_S12= {mag_ZCB_FB_1_S12:.4f}")
print(f"ang_ZCB_FB_1_S12= {ang_ZCB_FB_1_S12:.4f}°")
print("--------------------")
print(f"mag_ZCB_FB_1_S21= {mag_ZCB_FB_1_S21:.4f}")
print(f"ang_ZCB_FB_1_S21= {ang_ZCB_FB_1_S21:.4f}°")
print("--------------------")
print(f"mag_ZCB_FB_1_S22= {mag_ZCB_FB_1_S22:.4f}")
print(f"ang_ZCB_FB_1_S22= {ang_ZCB_FB_1_S22:.4f}°")
print("--------------------")
print("--------------------")
print("-----Z-parameters to S-parameters-----")
print("SCB_FB_1 = ({Z0[U]+[ZCB_FB_1]}**-1{Z0[U]-[ZCB_FB_1]})")
print("--------------------")
print("--------------------")
print("A_SCB_FB_1:")
print(A_SCB_FB_1)
print("B_SCB_FB_1:")
print(B_SCB_FB_1)
print("C_SCB_FB_1:")
print(C_SCB_FB_1)
print("SCB_FB_1:")
print(SCB_FB_1)
print("--------------------")
print("--rect to polar--")
print("--------------------")
print(f"mag_SCB_FB_1_S11= {mag_SCB_FB_1_S11:.4f}")
print(f"ang_SCB_FB_1_S11= {ang_SCB_FB_1_S11:.4f}°")
print("--------------------")
print(f"mag_SCB_FB_1_S12= {mag_SCB_FB_1_S12:.4f}")
print(f"ang_SCB_FB_1_S12= {ang_SCB_FB_1_S12:.4f}°")
print("--------------------")
print(f"mag_SCB_FB_1_S21= {mag_SCB_FB_1_S21:.4f}")
print(f"ang_SCB_FB_1_S21= {ang_SCB_FB_1_S21:.4f}°")
print("--------------------")
print(f"mag_SCB_FB_1_S22= {mag_SCB_FB_1_S22:.4f}")
print(f"ang_SCB_FB_1_S22= {ang_SCB_FB_1_S22:.4f}°")
print("--------------------")
print("-----Stability Analysis-----")
print(f"dl_SCB_FB_1= {dl_SCB_FB_1:.4f}")
print(f"mag_dl_SCB_FB_1= {mag_dl_SCB_FB_1:.4f}")
print(f"ang_dl_SCB_FB_1= {ang_dl_SCB_FB_1:.4f}°")
print(f"K_SCB_FB_1= {K_SCB_FB_1:.4f}")
print(f"mu_out_SCB_FB_1= {mu_out_SCB_FB_1:.4f}")
print(f"mu_in_SCB_FB_1= {mu_in_SCB_FB_1:.4f}")
print("--------------------")
print("-----Output Stability circles/Terminating Stability circle-----")
print(f"Ct= {Ct:.4f}")
print(f"Rt= {Rt:.4f}")
print("--rect to polar--")
print("--------------------")
print(f"mag_Ct= {mag_Ct:.4f}")
print(f"ang_Ct= {ang_Ct:.4f}°")
print("--------------------")
print(f"Gamma_T= {Gamma_T:.4f}")
print("--------------------")
print(f"Gamma_IN= {Gamma_IN:.4f}")
print(f"mag_Gamma_IN= {mag_Gamma_IN:.4f}")
print(f"ang_Gamma_IN= {ang_Gamma_IN:.4f}°")
print(f"Z_IN= {Z_IN:.4f} ohms")
print("--------------------")
print(f"Z_R= {Z_R:.4f} ohms")
print(f"Gamma_R= {Gamma_R:.4f}")
print(f"mag_Gamma_R= {mag_Gamma_R:.4f}")
print(f"ang_Gamma_R= {ang_Gamma_R:.4f}°")
print("-----IMN Design-Resonator Network-----")
print(f"theta_R2= {theta_R2:.4f}°")
print(f"theta_R2_2= {theta_R2_2:.4f}°")
print(f"theta_R1= {theta_R1:.4f}°")
print(f"theta_R1_2= {theta_R1_2:.4f}°")
print("-----OMN Design-Termination Network-----")
print(f"theta_T2= {theta_T2:.4f}°")
print(f"theta_T2_2= {theta_T2_2:.4f}°")
print(f"theta_T1= {theta_T1:.4f}°")
print(f"theta_T1_2= {theta_T1_2:.4f}°")
