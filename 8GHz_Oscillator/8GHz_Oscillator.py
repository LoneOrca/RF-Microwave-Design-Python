import numpy as np
import scipy
import skrf as rf
import matplotlib.pyplot as plt
#------Srinivas Nittala------
# *Oscillator using ATF-33143 GaAs FET @ 8GHz*
#----------------------------
#-----Part-1-----
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
f0 = 8*G
#---------------------------------
#- ".s2p" Just for refernce -
network = rf.Network("ATF33143_80.s2p")

freq = network.f
S = network.s

#---------------------------------
#----S-parameters---------
mag_S11 = 0.82
ang_S11 = 64.50

mag_S12 = 0.136
ang_S12 = -32.80

mag_S21 = 1.34
ang_S21 = -34.50

mag_S22 = 0.58
ang_S22 = 75.40

#----S-parameters (Polar to rect)----
S11 = (mag_S11*np.cos(np.deg2rad(ang_S11)))+1j*(mag_S11*np.sin(np.deg2rad(ang_S11)))
S12 = (mag_S12*np.cos(np.deg2rad(ang_S12)))+1j*(mag_S12*np.sin(np.deg2rad(ang_S12)))
S21 = (mag_S21*np.cos(np.deg2rad(ang_S21)))+1j*(mag_S21*np.sin(np.deg2rad(ang_S21)))
S22 = (mag_S22*np.cos(np.deg2rad(ang_S22)))+1j*(mag_S22*np.sin(np.deg2rad(ang_S22)))
# Common Source (CS)
SCS = np.array([
    [S11, S12],
    [S21, S22]
])

#---Stability Analysis---
dl = (S11*S22) - (S12*S21)
mag_dl = np.abs(dl)
ang_dl = np.angle(dl, deg=True)
K = (1-(mag_S11)**2-(mag_S22)**2+(mag_dl)**2)/(2*(mag_S12*mag_S21))

#-----Part-2-----
#----S-parameters to Y-parameters----
# YCS = Y0 ({[U]-[SCS]}{U]+[SCS]}**-1)
# YCS = Y-Parameters Common Source
U = np.eye(2)
A_S2Y = U - SCS
B_S2Y = U + SCS
B_S2Y_Inv = np.linalg.inv(B_S2Y)
YCS_S2Y = A_S2Y @ B_S2Y_Inv
YCS = Y0*(YCS_S2Y)
YCS_full_precision = YCS
#------------------------
#-----FET Common-Source Notation-----
YCS_ygg = YCS_full_precision[0, 0] # Gate-Gate
YCS_ygd = YCS_full_precision[0, 1] # Gate-Drain
YCS_ydg = YCS_full_precision[1, 0] # Drain-Gate
YCS_ydd = YCS_full_precision[1, 1] # Drain-Drain
#-----Part-2.1-----
#-----Conversion of 2x2 to 3x3 matrix[YCS_full_precision to Y_I]------------
YCS_ygs = - (YCS_ygg + YCS_ygd)
YCS_yds = - (YCS_ydg + YCS_ydd)
YCS_ysg = - (YCS_ygg + YCS_ydg)
YCS_ysd = - (YCS_ygd + YCS_ydd)
YCS_yss = - (YCS_ygs + YCS_yds)
YCS_yss_2 = - (YCS_ysg + YCS_ysd)
#-----Rectangular to Polar ------------
mag_YCS_ygg = np.abs(YCS_ygg)
ang_YCS_ygg = np.angle(YCS_ygg, deg = True)

mag_YCS_ygd = np.abs(YCS_ygd)
ang_YCS_ygd = np.angle(YCS_ygd, deg = True)

mag_YCS_ygs = np.abs(YCS_ygs)
ang_YCS_ygs = np.angle(YCS_ygs, deg = True)

mag_YCS_ydg = np.abs(YCS_ydg)
ang_YCS_ydg = np.angle(YCS_ydg, deg = True)

mag_YCS_ydd = np.abs(YCS_ydd)
ang_YCS_ydd = np.angle(YCS_ydd, deg = True)

mag_YCS_yds = np.abs(YCS_yds)
ang_YCS_yds = np.angle(YCS_yds, deg = True)

mag_YCS_ysg = np.abs(YCS_ysg)
ang_YCS_ysg = np.angle(YCS_ysg, deg = True)

mag_YCS_ysd = np.abs(YCS_ysd)
ang_YCS_ysd = np.angle(YCS_ysd, deg = True)

mag_YCS_yss = np.abs(YCS_yss)
ang_YCS_yss = np.angle(YCS_yss, deg = True)

mag_YCS_yss_2 = np.abs(YCS_yss_2)
ang_YCS_yss_2 = np.angle(YCS_yss_2, deg = True)
#-----Part-2.2-----
YI = np.array([
    [YCS_ygg, YCS_ygd, YCS_ygs ], #GATE
    [YCS_ydg, YCS_ydd, YCS_yds ], #DRAIN
    [YCS_ysg, YCS_ysd, YCS_yss ]  #SOURCE
])
#-----Part-2.3-----
# Let's reduce the 3x3 matrix to 2x2 {[YI] to [YCB]}

YCG = np.array([
    [YCS_yss, YCS_ysd ],
    [YCS_yds, YCS_ydd ]
])
#-----Part-3-----
#----Y-parameters to S-parameters----
# SCG = ({[U]Y0-[YCG]}{[U]Y0+[YCG]}**-1)
A_YCG = U*Y0
B_YCG = A_YCG - YCG
C_YCG = A_YCG + YCG
D_YCG = np.linalg.inv(C_YCG)
SCG = B_YCG @ D_YCG
#------------------------
SCG_S11 = SCG[0, 0]
SCG_S12 = SCG[0, 1]
SCG_S21 = SCG[1, 0]
SCG_S22 = SCG[1, 1]
#-----Rectangular to Polar ------------
mag_SCG_S11 = np.abs(SCG_S11)
ang_SCG_S11 = np.angle(SCG_S11, deg = True)

mag_SCG_S12 = np.abs(SCG_S12)
ang_SCG_S12 = np.angle(SCG_S12, deg = True)

mag_SCG_S21 = np.abs(SCG_S21)
ang_SCG_S21 = np.angle(SCG_S21, deg = True)

mag_SCG_S22 = np.abs(SCG_S22)
ang_SCG_S22 = np.angle(SCG_S22, deg = True)
#-----Part-3.1-----
#---SCB-Stability Analysis---
dl_SCG = (SCG_S11*SCG_S22) - (SCG_S12*SCG_S21)
mag_dl_SCG = np.abs(dl_SCG)
ang_dl_SCG = np.angle(dl_SCG, deg = True)
K_SCG = (1-(mag_SCG_S11)**2-(mag_SCG_S22)**2+(mag_dl_SCG)**2)/(2*(mag_SCG_S12*mag_SCG_S21))
#-----Part-4-----
#----S-parameters to Z-parameters----
# ZCG = (Z0{[U]+[SCG]}{[U]-[SCG]}**-1)
A_ZCG = U + SCG
B_ZCG = U - SCG
C_ZCG = np.linalg.inv(B_ZCG)
D_ZCG = A_ZCG @ C_ZCG
ZCG_precison = Z0*D_ZCG
ZCG = ZCG_precison
#------------------------
ZCG_S11 = ZCG[0, 0]
ZCG_S12 = ZCG[0, 1]
ZCG_S21 = ZCG[1, 0]
ZCG_S22 = ZCG[1, 1]
#-----Rectangular to Polar ------------
mag_ZCG_S11 = np.abs(ZCG_S11)
ang_ZCG_S11 = np.angle(ZCG_S11, deg = True)

mag_ZCG_S12 = np.abs(ZCG_S12)
ang_ZCG_S12 = np.angle(ZCG_S12, deg = True)

mag_ZCG_S21 = np.abs(ZCG_S21)
ang_ZCG_S21 = np.angle(ZCG_S21, deg = True)

mag_ZCG_S22 = np.abs(ZCG_S22)
ang_ZCG_S22 = np.angle(ZCG_S22, deg = True)
#-----Part-4.1-----
#-Add Gate Feedback Reactance (X_G)-
X_G = np.linspace(-100, 100, 1000) 
# we can keep 9 or 1000, if we keep 1000 then graph will be smooth with no spikes.
# for smooth spike in the graph i took -500 to 500 (or) -100 to 100
FG = np.array([
    [1, 1],
    [1, 1]
])
ZCG_FG = np.zeros((len(X_G), 2, 2), dtype=complex)
for i, XG in enumerate(X_G):
   ZCG_FG[i] = ZCG + 1j * XG * FG
S_CG_FG = np.zeros((len(X_G), 2, 2), dtype=complex)
for i in range(len(X_G)):
    Z = ZCG_FG[i]
    S_CG_FG[i] = (Z - Z0 * U) @ np.linalg.inv(Z + Z0 * U)
#----magnitude of the S-parameters----
mag_S11_S_CG_FG = np.abs(S_CG_FG[:, 0, 0])
mag_S12_S_CG_FG = np.abs(S_CG_FG[:, 0, 1])
mag_S21_S_CG_FG = np.abs(S_CG_FG[:, 1, 0])
mag_S22_S_CG_FG = np.abs(S_CG_FG[:, 1, 1])
#-----Part-4.2-----
#----Stability calculations----
#----Delta Values----
delta = (
    S_CG_FG[:, 0, 0] * S_CG_FG[:, 1, 1]
    - S_CG_FG[:, 0, 1] * S_CG_FG[:, 1, 0]
)
mag_delta_S_CG_FG = np.abs(delta)
#-----Stability Factor K-----

K_S_CG_FG = (
    1
    - mag_S11_S_CG_FG**2
    - mag_S22_S_CG_FG**2
    + mag_delta_S_CG_FG**2
) / (
    2 * mag_S12_S_CG_FG * mag_S21_S_CG_FG
)
#-----Part-4.3-----
#-----Mu Stability Factors-----

S11_FG = S_CG_FG[:, 0, 0]
S12_FG = S_CG_FG[:, 0, 1]
S21_FG = S_CG_FG[:, 1, 0]
S22_FG = S_CG_FG[:, 1, 1]

mu_out = (
    1 - mag_S11_S_CG_FG**2
) / (
    np.abs(S22_FG - delta * np.conj(S11_FG))
    + np.abs(S12_FG * S21_FG)
)

mu_in = (
    1 - mag_S22_S_CG_FG**2
) / (
    np.abs(S11_FG - delta * np.conj(S22_FG))
    + np.abs(S12_FG * S21_FG)
)
#-----Part-4.4-----
#-----Plot K,Mu_in and Mu_out versus Feedback Reactance-----
plt.figure()

plt.plot(X_G, K_S_CG_FG, 'g', linewidth=3, label='K')
plt.plot(X_G, mu_in, 'r', linewidth=3, label='Mu_in')
plt.plot(X_G, mu_out, 'b', linewidth=3, label='Mu_out')

plt.xlabel("Feedback Reactance (Ohm)")
plt.ylabel("Stability Factors")
plt.title("Stability Factors versus Feedback Reactance")

plt.xlim(-100, 100)

plt.grid(True)
plt.legend()
plt.show()
#-----Plot |S11| and |S22| versus Feedback Reactance-----
plt.figure()

plt.plot(X_G, mag_S11_S_CG_FG, 'r', linewidth=3, label='|S11|')
plt.plot(X_G, mag_S22_S_CG_FG, 'b', linewidth=3, label='|S22|')

plt.xlabel("Feedback Reactance (Ohm)")
plt.ylabel("Magnitude")
plt.title("|S11| and |S22| versus Feedback Reactance")
# That sharp dip or spike you are seeing right @ S11 is the X_G_1
plt.xlim(-100, 100)
plt.ylim(0, 6)

plt.grid(True)
plt.legend()
plt.show()


#-----Part-5-----
#------------------------
#- Rearrange the Feedback Reactance (X_G_1 = -64 ohms/-63.7638{parallax error})-
#idx = np.argmax(mag_S11_S_CG_FG)
X_G_1 = -64
CG = 1/(2*np.pi*(8*G)*np.abs(X_G_1))
theta_y = np.degrees(np.arctan(X_G_1 / Z0))
theta_x = theta_y % 180
ZCG_FG_1 = ZCG + 1j * X_G_1 * FG
#------------------------
ZCG_FG_1_S11 = ZCG_FG_1[0, 0]
ZCG_FG_1_S12 = ZCG_FG_1[0, 1]
ZCG_FG_1_S21 = ZCG_FG_1[1, 0]
ZCG_FG_1_S22 = ZCG_FG_1[1, 1]
#-----Rectangular to Polar ------------
mag_ZCG_FG_1_S11 = np.abs(ZCG_FG_1_S11)
ang_ZCG_FG_1_S11 = np.angle(ZCG_FG_1_S11, deg = True)

mag_ZCG_FG_1_S12 = np.abs(ZCG_FG_1_S12)
ang_ZCG_FG_1_S12 = np.angle(ZCG_FG_1_S12, deg = True)

mag_ZCG_FG_1_S21 = np.abs(ZCG_FG_1_S21)
ang_ZCG_FG_1_S21 = np.angle(ZCG_FG_1_S21, deg = True)

mag_ZCG_FG_1_S22 = np.abs(ZCG_FG_1_S22)
ang_ZCG_FG_1_S22 = np.angle(ZCG_FG_1_S22, deg = True)
#-----Part-5.1-----
#----Z-parameters to S-parameters----
# SCG_FG_1 = ({Z0[U]+[ZCG_FG_1]}**-1{Z0[U]-[ZCG_FG_1]})
A_SCG_FG_1 = ZCG_FG_1 + Z0*U
B_SCG_FG_1 = np.linalg.inv(A_SCG_FG_1)
C_SCG_FG_1 = ZCG_FG_1 - Z0*U
SCG_FG_1_precison = B_SCG_FG_1 @ C_SCG_FG_1
SCG_FG_1 = SCG_FG_1_precison
#------------------------
SCG_FG_1_S11 = SCG_FG_1[0, 0]
SCG_FG_1_S12 = SCG_FG_1[0, 1]
SCG_FG_1_S21 = SCG_FG_1[1, 0]
SCG_FG_1_S22 = SCG_FG_1[1, 1]
#-----Rectangular to Polar ------------
mag_SCG_FG_1_S11 = np.abs(SCG_FG_1_S11)
ang_SCG_FG_1_S11 = np.angle(SCG_FG_1_S11, deg = True)

mag_SCG_FG_1_S12 = np.abs(SCG_FG_1_S12)
ang_SCG_FG_1_S12 = np.angle(SCG_FG_1_S12, deg = True)

mag_SCG_FG_1_S21 = np.abs(SCG_FG_1_S21)
ang_SCG_FG_1_S21 = np.angle(SCG_FG_1_S21, deg = True)

mag_SCG_FG_1_S22 = np.abs(SCG_FG_1_S22)
ang_SCG_FG_1_S22 = np.angle(SCG_FG_1_S22, deg = True)
#-----Part-5.2-----
#----Stability calculations----
#----Delta Values----
dl_SCG_FG_1 = (SCG_FG_1_S11*SCG_FG_1_S22)-(SCG_FG_1_S12*SCG_FG_1_S21)
mag_dl_SCG_FG_1 = np.abs(dl_SCG_FG_1)
ang_dl_SCG_FG_1 = np.angle(dl_SCG_FG_1, deg = True)
K_SCG_FG_1 = (1-(mag_SCG_FG_1_S11)**2-(mag_SCG_FG_1_S22)**2+(mag_dl_SCG_FG_1)**2)/(2*(mag_SCG_FG_1_S12*mag_SCG_FG_1_S21))

mu_out_SCG_FG_1 = (
1 - mag_SCG_FG_1_S11**2
) / (
    np.abs(SCG_FG_1_S22 - dl_SCG_FG_1 * np.conj(SCG_FG_1_S11))
    + np.abs(SCG_FG_1_S12 * SCG_FG_1_S21)
)

mu_in_SCG_FG_1 = (
    1 - mag_SCG_FG_1_S22**2
) / (
    np.abs(SCG_FG_1_S11 - dl_SCG_FG_1 * np.conj(SCG_FG_1_S22))
    + np.abs(SCG_FG_1_S12 * SCG_FG_1_S21))
#-----Part-5.3-----
#----Output Stability circles/Terminating Stability circle----    
A_Ct = (SCG_FG_1_S22 - (dl_SCG_FG_1)*np.conj(SCG_FG_1_S11))
B_Ct = np.conj(A_Ct)
Ct = B_Ct/((mag_SCG_FG_1_S22)**2-(mag_dl_SCG_FG_1)**2)  

Rt = np.abs((SCG_FG_1_S12*SCG_FG_1_S21)/((mag_SCG_FG_1_S22)**2-(mag_dl_SCG_FG_1)**2) )
#-----Rectangular to Polar ------------
mag_Ct = np.abs(Ct)
ang_Ct = np.angle(Ct, deg = True)
#------------------------
# An|S11| dictates that the Smith Chart origin is unstable. 
# If this origin lies inside the perimeter of the stability circle, "the entire zone enclosed by the circle becomes the unstable region."
# SO i choose Gamma_T from the circle.
#------------------------
mag_Gamma_T = 0.5000
ang_Gamma_T = -140.0000
Gamma_T = (mag_Gamma_T*np.cos(np.deg2rad(ang_Gamma_T)))+1j*(mag_Gamma_T*np.sin(np.deg2rad(ang_Gamma_T)))

Gamma_IN = (SCG_FG_1_S11)+((SCG_FG_1_S12*SCG_FG_1_S21*Gamma_T)/(1-SCG_FG_1_S22*Gamma_T))
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
#----OMN Design-Termination Network----
dT2_lambda = 0.0280
theta_T2 = 360*dT2_lambda
theta_T2_2 = theta_T2*2
dT1_lambda = 0.1330
theta_T1 = 360* dT1_lambda
theta_T1_2 = theta_T1*2
#----IMN Design-Resonator Network----
dR2_lambda = 0.4442
theta_R2 = 360*dR2_lambda
theta_R2_2 = theta_R2*2
dR1_lambda = 0.0608
theta_R1 = 360*dR1_lambda
theta_R1_2 = theta_R1*2
#--------Display values--------
print("Device:Oscillator using ATF-33143 GaAs FET @ 8GHz")
print("Center Frequency: 8GHz")
print(f"S11= {S11:.4f}")
print(f"S12= {S12:.4f}")
print(f"S21= {S21:.4f}")
print(f"S22= {S22:.4f}")
print(f"SCS=")
print(np.round(SCS,4))
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
print(f"YCS_S2Y=")
print(np.round(YCS_S2Y,4))
print(f"YCS=")
print(np.round(YCS,4))
print("YCS full precision:")
print(YCS_full_precision)
print("--------------------")
print("--YI matrix elements--")
print("--------------------")
print(f"YCS_ygg= {YCS_ygg:.4f}")
print(f"YCS_ygd= {YCS_ygd:.4f}")
print(f"YCS_ygs= {YCS_ygs:.4f}")
print(f"YCS_ydg= {YCS_ydg:.4f}")
print(f"YCS_ydd= {YCS_ydd:.4f}")
print(f"YCS_yds= {YCS_yds:.4f}")
print(f"YCS_ysg= {YCS_ysg:.4f}")
print(f"YCS_ysd= {YCS_ysd:.4f}")
print(f"YCS_yss= {YCS_yss:.4f}")
print("--------------------")
print("--rect to polar--")
print("--------------------")
print(f"mag_YCS_ygg= {mag_YCS_ygg*k:.4f}")
print(f"ang_YCS_ygg= {ang_YCS_ygg:.4f}°")
print("--------------------")
print(f"mag_YCS_ygd= {mag_YCS_ygd*k:.4f}")
print(f"ang_YCS_ygd= {ang_YCS_ygd:.4f}°")
print("--------------------")
print(f"mag_YCS_ygs= {mag_YCS_ygs*k:.4f}")
print(f"ang_YCS_ygs= {ang_YCS_ygs:.4f}°")
print("--------------------")
print(f"mag_YCS_ydg= {mag_YCS_ydg*k:.4f}")
print(f"ang_YCS_ydg= {ang_YCS_ydg:.4f}°")
print("--------------------")
print(f"mag_YCS_ydd= {mag_YCS_ydd*k:.4f}")
print(f"ang_YCS_ydd= {ang_YCS_ydd:.4f}°")
print("--------------------")
print(f"mag_YCS_yds= {mag_YCS_yds*k:.4f}")
print(f"ang_YCS_yds= {ang_YCS_yds:.4f}°")
print("--------------------")
print(f"mag_YCS_ysg= {mag_YCS_ysg*k:.4f}")
print(f"ang_YCS_ysg= {ang_YCS_ysg:.4f}°")
print("--------------------")
print(f"mag_YCS_ysd= {mag_YCS_ysd*k:.4f}")
print(f"ang_YCS_ysd= {ang_YCS_ysd:.4f}°")
print("--------------------")
print(f"mag_YCS_yss= {mag_YCS_yss*k:.4f}")
print(f"ang_YCS_yss= {ang_YCS_yss:.4f}°")
print("--------------------")
print(f"mag_YCS_yss_2= {mag_YCS_yss_2:.4f}")
print(f"ang_YCS_yss_2= {ang_YCS_yss_2:.4f}°")
print("--------------------")
print(np.isclose(YCS_yss, YCS_yss_2))
print("--YI matrix --")
print("YI:")
print(YI)
print("--------------------")
print("--Let's reduce the 3x3 matrix to 2x2 {[YI] to [YCB]} --")
print("--YCB matrix --")
print("YCG:")
print(YCG)
print("--------------------")
print("--------------------")
print("-----Y-parameters to S-parameters-----")
print("SCG = ({[U]Y0-[YCG]}{[U]Y0+[YCG]}**-1)")
print("--------------------")
print("A_YCG:")
print(A_YCG)
print("B_YCG:")
print(B_YCG)
print("C_YCG:")
print(C_YCG)
print("D_YCG:")
print(D_YCG)
print("SCG:")
print(SCG)
print("--------------------")
print("--rect to polar--")
print("--------------------")
print(f"mag_SCG_S11= {mag_SCG_S11:.4f}")
print(f"ang_SCG_S11= {ang_SCG_S11:.4f}°")
print("--------------------")
print(f"mag_SCG_S12= {mag_SCG_S12:.4f}")
print(f"ang_SCG_S12= {ang_SCG_S12:.4f}°")
print("--------------------")
print(f"mag_SCG_S21= {mag_SCG_S21:.4f}")
print(f"ang_SCG_S21= {ang_SCG_S21:.4f}°")
print("--------------------")
print(f"mag_SCG_S22= {mag_SCG_S22:.4f}")
print(f"ang_SCG_S22= {ang_SCG_S22:.4f}°")
print("--------------------")
print("-----SCB-Stability Analysis-----")
print(f"dl_SCG= {dl_SCG:.4f}")
print(f"mag_dl_SCG= {mag_dl_SCG:.4f}")
print(f"ang_dl_SCG= {ang_dl_SCG:.4f}")
print(f"K_SCG= {K_SCG:.4f}")
if mag_dl_SCG<1 and K_SCG>1:
 print("Stability:Unconditionally Stable")
else:
 print("Stability: UnStable")
print("--------------------")
print("-----S-parameters to Z-parameters-----")
print("ZCG = (Z0{[U]+[SCG]}{[U]-[SCG]}**-1)")
print("--------------------")
print("A_ZCG:")
print(A_ZCG)
print("B_ZCG:")
print(B_ZCG)
print("C_ZCG:")
print(C_ZCG)
print("D_ZCG:")
print(D_ZCG)
print("ZCG:")
print(ZCG)
print("--------------------")
print("--rect to polar--")
print("--------------------")
print(f"mag_ZCG_S11= {mag_ZCG_S11:.4f}")
print(f"ang_ZCG_S11= {ang_ZCG_S11:.4f}°")
print("--------------------")
print(f"mag_ZCG_S12= {mag_ZCG_S12:.4f}")
print(f"ang_ZCG_S12= {ang_ZCG_S12:.4f}°")
print("--------------------")
print(f"mag_ZCG_S21= {mag_ZCG_S21:.4f}")
print(f"ang_ZCG_S21= {ang_ZCG_S21:.4f}°")
print("--------------------")
print(f"mag_ZCG_S22= {mag_ZCG_S22:.4f}")
print(f"ang_ZCG_S22= {ang_ZCG_S22:.4f}°")
print("--------------------")
print("-----Display Feedback Impedance Matrices-----")
for i, XG in enumerate(X_G):
    print(f"\nFeedback Reactance = {XG} Ohms")
    print("ZCG_FG =")
    print(ZCG_FG[i])
print("--------------------")
print("-----Display S-Parameter Matrices-----")
for i, XG in enumerate(X_G):
    print(f"\nX_G = {XG:.1f} Ohm")
    print(S_CG_FG[i])
print("--------------------")
print("-----Display S-Parameter Magnitudes-----")
for i, XG in enumerate(X_G):
    print(f"\nX_G = {XG:.1f} Ohm")
    print(f"|S11| = {mag_S11_S_CG_FG[i]:.4f}")
    print(f"|S12| = {mag_S12_S_CG_FG[i]:.4f}")
    print(f"|S21| = {mag_S21_S_CG_FG[i]:.4f}")
    print(f"|S22| = {mag_S22_S_CG_FG[i]:.4f}")

print("-----Display Delta-----")
print("--------------------")
for i, XG in enumerate(X_G):
    print(f"X_G = {XG:.1f} Ohm, Delta = {delta[i]}")
print("--------------------")
print("-----Display Magnitude of Delta-----")
for i, XG in enumerate(X_G):
    print(f"X_G = {XG:.1f} Ohm, |Delta| = {mag_delta_S_CG_FG[i]:.4f}")
print("--------------------")
print("-----Display Stability Factor K-----")

for i, XG in enumerate(X_G):
    print(f"X_G = {XG:.1f} Ohm, K = {K_S_CG_FG[i]:.4f}")
print("--------------------")
print("-----Display Mu Stability Factors-----")
for i, XG in enumerate(X_G):
    print(f"\nX_G = {XG:.1f} Ohm")
    print(f"Mu_in  = {mu_in[i]:.4f}")
    print(f"Mu_out = {mu_out[i]:.4f}")
print("--------------------")
print("--------------------")
print("-----Rearrange the Feedback Reactance-----")
print("-----That sharp dip or spike you are seeing right @ S11 is the X_B_1-----")
#print(f"Optimum XG = {X_G[idx]:.4f} ohms")
#print(f"Maximum |S11| = {mag_S11_S_CG_FG[idx]:.4f}")
print(f"X_G_1= {X_G_1:.4f} ohms")
print(f"CG = {CG*1e12:.4f} pF")
print(f"Theta_x = {theta_x:.4f}°")
print("ZCG_FG_1:")
print(ZCG_FG_1)
print("--------------------")
print("--rect to polar--")
print("--------------------")
print(f"mag_ZCG_FG_1_S11= {mag_ZCG_FG_1_S11:.4f}")
print(f"ang_ZCG_FG_1_S11= {ang_ZCG_FG_1_S11:.4f}°")
print("--------------------")
print(f"mag_ZCG_FG_1_S12= {mag_ZCG_FG_1_S12:.4f}")
print(f"ang_ZCG_FG_1_S12= {ang_ZCG_FG_1_S12:.4f}°")
print("--------------------")
print(f"mag_ZCG_FG_1_S21= {mag_ZCG_FG_1_S21:.4f}")
print(f"ang_ZCG_FG_1_S21= {ang_ZCG_FG_1_S21:.4f}°")
print("--------------------")
print(f"mag_ZCG_FG_1_S22= {mag_ZCG_FG_1_S22:.4f}")
print(f"ang_ZCG_FG_1_S22= {ang_ZCG_FG_1_S22:.4f}°")
print("--------------------")
print("--------------------")
print("-----Z-parameters to S-parameters-----")
print("SCB_FB_1 = ({Z0[U]+[ZCB_FB_1]}**-1{Z0[U]-[ZCB_FB_1]})")
print("--------------------")
print("--------------------")
print("A_SCG_FG_1:")
print(A_SCG_FG_1)
print("B_SCG_FG_1:")
print(B_SCG_FG_1)
print("C_SCG_FG_1:")
print(C_SCG_FG_1)
print("SCG_FG_1:")
print(SCG_FG_1)
print("--------------------")
print("--rect to polar--")
print("--------------------")
print(f"mag_SCG_FG_1_S11= {mag_SCG_FG_1_S11:.4f}")
print(f"ang_SCG_FG_1_S11= {ang_SCG_FG_1_S11:.4f}°")
print("--------------------")
print(f"mag_SCG_FG_1_S12= {mag_SCG_FG_1_S12:.4f}")
print(f"ang_SCG_FG_1_S12= {ang_SCG_FG_1_S12:.4f}°")
print("--------------------")
print(f"mag_SCG_FG_1_S21= {mag_SCG_FG_1_S21:.4f}")
print(f"ang_SCG_FG_1_S21= {ang_SCG_FG_1_S21:.4f}°")
print("--------------------")
print(f"mag_SCG_FG_1_S22= {mag_SCG_FG_1_S22:.4f}")
print(f"ang_SCG_FG_1_S22= {ang_SCG_FG_1_S22:.4f}°")
print("--------------------")
print("-----Stability Analysis-----")
print(f"dl_SCG_FG_1= {dl_SCG_FG_1:.4f}")
print(f"mag_dl_SCG_FG_1= {mag_dl_SCG_FG_1:.4f}")
print(f"ang_dl_SCG_FG_1= {ang_dl_SCG_FG_1:.4f}°")
print(f"K_SCG_FG_1= {K_SCG_FG_1:.4f}")
print(f"mu_out_SCG_FG_1= {mu_out_SCG_FG_1:.4f}")
print(f"mu_in_SCG_FG_1= {mu_in_SCG_FG_1:.4f}")
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
print("-----OMN Design-Termination Network-----")
print(f"theta_T2= {theta_T2:.4f}°")
print(f"theta_T2_2= {theta_T2_2:.4f}°")
print(f"theta_T1= {theta_T1:.4f}°")
print(f"theta_T1_2= {theta_T1_2:.4f}°")
print("-----IMN Design-Resonator Network-----")
print(f"theta_R2= {theta_R2:.4f}°")
print(f"theta_R2_2= {theta_R2_2:.4f}°")
print(f"theta_R1= {theta_R1:.4f}°")
print(f"theta_R1_2= {theta_R1_2:.4f}°")
