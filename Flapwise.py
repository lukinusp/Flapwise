#*******************************************************************************
#
#Nikolas Lukin
#5381328
#Trabalho de formatura
#
#*******************************************************************************

# Bibliotecas

from functools import reduce
from operator import concat
import sys
import pygame
from pygame.locals import *
import math
import numpy as np
import scipy
import scipy.linalg as la
import matplotlib.pyplot as plt
import time
import scipy.fftpack
from scipy.signal import savgol_filter
import pylab as pl
from scipy import interpolate
from tabulate import tabulate
from statistics import NormalDist
import random

start_time = time.time()

# Props GeomÃ©tricas da pÃ¡

L    = 86.37                            #[m]   - Comprimento da pÃ¡
Iy   = 165*10**(-3)                     #[m^4] - Momento de inÃ©rcia 
th = 0.857                              #[m]   - Espessura equivalente
ar = 1.35*10**11                        #[Pa]  - Parâmetro de rigidez
br = -0.064                             #[-]
am = 1050
bm = -0.027
Cmax = 6.8
bc = -0.04
ac = -Cmax/(L**2)
Tilt = 5 * math.pi / 180                #[°] - Tilt angle
Cone = 4.5 * math.pi / 180              #[°] - Cone angle
offset = 1.4

# Props da Torre

d_T = 4.0                               # [m] - Diam no TH
D_T = 8.0                              # [m] - Diam na base da torre
H = 150.0                               # [m] - Altura da torre
Overhang = 7.07                         # [m] - Overhang
Slope_T = (D_T - d_T)/H
U_max = Overhang + L*math.sin(Cone + Tilt) - \
    (D_T - Slope_T * (H - L*math.cos(Cone + Tilt)))

# Props MecÃ¢nicas

E   = 40*10**9                          #[Pa]     - MÃ³dulo de Young
g   = 9.81                              #[m/s^2]  - Gravidade
Rho = 1.225                             #[kg/m^3] - Densidade do ar padrÃ£o (IEC61400)

# ParÃ¢metros da simulaÃ§Ã£o

Tf = 120                                #[s]   - Tempo final da simulaÃ§Ã£o
n = 25                                  #[ ]   - NÃºmero de nÃ³s da malha
nsamp = 1
h = float(L/n)                          #[m]   - Passo da malha
k = 0.92*h*math.pi/math.sqrt(E/340)     #[s]   - Passo de tempo
nt = int(Tf/k)+1                        #[ ]   - NÃºmero de iteraÃ§Ãµes da simulaÃ§Ã£o
au = 45                                 #[ ]   - Passos para atualizar matriz de rigidez
ta = 90                                 #[s]   - Duração da amostragem do sinal
factor = 50
factor2 = 0
Phy  = [0]*(nt)#[0.5*math.pi]*(nt)#[0]*(nt)      #[rad] - Ã‚ngulo da pÃ¡

# ParÃ¢metros do vento

Iref = 0.16
A1 = 42
Vmin = 10                                #[m/s]    - Velocidade do vento
Vci = 4
Vr = 11.4
Vco = 20
Vmax = 11
Vref = 50
DesV = Iref * (0.75*Vr + 5.6)
Hhub = 150
Turb = 0
Tg = 10.5
Tig = 60
PV = 1
CisCoeff = 0.2
inc = 0
VelV = np.arange(Vmin, Vmax, PV)

# ParÃ¢metros de controle

Rn = 9.5
TSR = 7.5                               #[]       - Tip Speed Ratio
Rot_max = (Rn*2*3.1415)/60              #[rpm]    - RotaÃ§Ã£o mÃ¡xima da pÃ¡
control_rate = 100
alpha_max = 3 * math.pi / 180
alpha_min = -alpha_max
alpha_rate_max = 8 * math.pi / 180
alpha_rate_min = -alpha_rate_max
nc = int( 1 / (control_rate * k))
Kp = 0#0.055
Kd = 0#31.484
Ki = 0#0.053
d_alpha = 0
Sigma_old = 0
err_dot = 0
err_int = 0

# InicializaÃ§Ã£o

x        = np.arange(0,L,h)             #[m]      - Dist da raÃ­z da pÃ¡
t        = np.arange(0,Tf,k)            #[s]      - Vetor do tempo
l        = [0]*n                        #[kg/s^2] - Gerador de linha
K       = []                            # Matriz de rigidez
Sigma    = np.zeros(nt)                 #[Pa]     - TensÃ£o na raiz da pÃ¡
Empuxo    = np.zeros(nt)                #[Pa]     - Empuxo na raiz da pÃ¡
Momento    = np.zeros(nt)               #[Pa]     - Momento na raiz da pÃ¡
alpha    = np.zeros(nt)                 #[Pa]     - Ângulo de pitch
U0p_new    = [0]*(2*(n-1))              #[m]      - Desloc do nÃ³ (futuro)
U0p_old    = [0]*(2*(n-1))              #[m]      - Desloc do nÃ³ (presente)
Uw        = [0]*(n-1)                   #[m/s] - Vel vento no nÃ³ (futuro)
Uhub        = [0]*(nt)                  #[m/s] - Vel vento no nÃ³ (futuro)
Q        = [0.0]*(n-1)                  #[kg/s^2] - Carregamento no nÃ³
f        = [0.0]*(n-1)
U_ref    = [0.0]*(n-1)
Q_ref        = np.ones(n-1)
b        = [0]*n 
chord = [0]*n
EIy = [0]*n
d1EI = [0.0] * n
d2EI = [0.0] * n
m = am * np.exp(np.arange(0,L,h) * bm)
mc = 460 * np.ones(n)
EIy = ar * np.exp(np.arange(0,L,h) * br)
EIy1 = ar * np.exp(np.arange(0,L,h) * 1.05*br)
EIyc = 1.86*(10**10) * np.ones(n)
chordc = 4.05 * np.ones(n)
twist = [0]*n

for i in range(len(x)):
    chord[i] = np.polyval([ac,0,Cmax], x[i])
    b[i] = 0.043*math.sqrt(384*EIy[i]/(17*L**3)) * chord[i]
bcc = 0.043*math.sqrt(384*EIyc[0]/(17*L**3))*chordc[0] * np.ones(n)

Bs = np.diag(np.full(n,b))        # Matriz de amortecimento
Ms = np.diag(np.full(n,m))        # Matriz de massa
Bs = np.delete(Bs, 0, 1)
Bs = np.delete(Bs, 0, 0)
Ms = np.delete(Ms, 0, 1)
Ms = np.delete(Ms, 0, 0)

# Variáveis de animação e gráficos

rate = int(0.1/k)
Lcolor = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
mycmap = plt.colormaps['hot_r']

# Variáveis de análise

Max_Fx =    [[0 for i in range(nsamp)] for j in range(int((Vmax-Vmin)/PV + 1))]
Max_My =    [[0 for i in range(nsamp)] for j in range(int((Vmax-Vmin)/PV + 1))]
Max_U =     [[0 for i in range(nsamp)] for j in range(int((Vmax-Vmin)/PV + 1))]
Med_U =     [[0 for i in range(nsamp)] for j in range(int((Vmax-Vmin)/PV + 1))]
Max_Sigma = [[0 for i in range(nsamp)] for j in range(int((Vmax-Vmin)/PV + 1))]
Max_Alpha = [[0 for i in range(nsamp)] for j in range(int((Vmax-Vmin)/PV + 1))]
Sigma_rms = [[0 for i in range(nsamp)] for j in range(int((Vmax-Vmin)/PV + 1))]
Sigma_n1 =  [[0 for i in range(nsamp)] for j in range(int((Vmax-Vmin)/PV + 1))]

LeadAngle = np.arange(-0.5*math.pi/9, math.pi/9, 0.005*math.pi) 
Rot = [0]*len(VelV)
Q_comp = [0]*len(VelV)
Q_ref2 = [0]*len(VelV)

# Polares NACA 0012

def Cl(x, y, L):
    #Cl = -53.56*(x**3) + 8.9223*(x**2) + 7.1829*x + 0.3435
    Cl = (-16.753*(x**3) + 1.1562*(x**2) + 5.1624*x + 0.3274)
    return Cl

def Cd(x):
    Cd = 21.352*(x**4) - 6.796*(x**3) + 0.4795*(x**2) + 0.0431*x + 0.0078
    return Cd

def CalcA(EIy, h):
    A  = (EIy)/(h**4)
    return A

def CalcB(m, g, L, x, Phy, Tilt, Cone, Rot, h, d2I):
    B  = (m*g*(L - x)*math.cos(Phy)*math.cos(Tilt)*math.cos(Cone) \
    - 0.5*m*(Rot**2)*(L**2-x**2)*(math.cos(Cone)**2) + d2I)/(12*(h**2))
    return B

def CalcC(m, Rot, x, g, Phy, Tilt, Cone, h):
    C  = (m*(Rot**2)*x-m*g*math.cos(Phy)*math.cos(Tilt))*math.cos(Cone)/(12*h)
    return C

def CalcD(d1I, h):
    D  = (d1I)/(4*(h**3))
    return D

def CalcQ(Rho, chord, Uw, Rot, x, m, g, Theta, Cone, Phy, Tilt, h):
    Q = -0.5*Rho*chord*(Uw**2 + (Rot * x)**2)*(math.cos(Theta)*Cl(Theta, x, L)\
    +math.sin(Theta)*Cd(Theta))-m*g*math.sin(Cone)*math.cos(Phy)*math.sin(Tilt)
    return Q

def CalcTheta(Rot, Uw, inc, x, twist):
    if Rot > 0:
        Theta = math.atan( Uw * np.abs( math.cos(inc) ) / ( Rot * x ) ) \
        - twist
    else:
        Theta = math.pi/2.0 - twist
    return Theta

def PosTable(Chan, Vmax, Vmin, PV):
    data = []
    r=0
    for Vhub in range(Vmax, Vmin, -PV):
        data.append([Vhub, "{:.3f}".format(np.min(Chan[:][r])), \
                           "{:.3f}".format(np.average(Chan[:][r])),  \
                           "{:.3f}".format(np.max(Chan[:][r])), \
                           "{:.3f}".format(np.std(Chan[:][r]))])
        r += 1
        
    headers=["Vento [m/s]", "Min", "Med", "Max", "DesvPad"]
    print (tabulate(data, headers))
    #print(pandas.DataFrame(data, headers, headers))
    print('                                         ')
    print('-----------------------------------------')
    print('                                         ')
    
    return None

def Input(infile, x):
    
    xqr,yqr = [], []
    with open(infile,'r') as file:
        for line in file:
            row = line.split()
            xqr.append(float(row[0]))
            yqr.append(float(row[1]))
    xqr = np.array(xqr)
    yqr = np.array(yqr)
    tck = interpolate.splrep(xqr, yqr)
    yqr = interpolate.splev(x, tck)
    return yqr

def Input2(infile):
    
    xqr,yqr = [], []
    with open(infile,'r') as file:
        for line in file:
            row = line.split()
            xqr.append(float(row[0]))
            yqr.append(float(row[1]))
    return xqr, yqr

# Leitura dos parâmetros da DTU 10MW

chord = Input('Chord.txt', x)
chord[-1] = 0.1
chord[-2] = 1.5
CDx, CDy = Input2('CD.txt')
CLx, CLy = Input2('CL.txt')
if Vmin < 10:
    Q_ref2[-1] = Input('q10.txt', x)
    if Vmin < 9:
        Q_ref2[-2] = Input('q9.txt', x)
        if Vmin < 8:
            Q_ref2[-3] = Input('q8.txt', x)
            if Vmin < 7:
                Q_ref2[-4] = Input('q7.txt', x)
                if Vmin < 6:
                    Q_ref2[-5] = Input('q6.txt', x)
twist = (math.pi/180) * Input('Twist.txt', x)
m = Input('Mass2.txt', x)
ysref = Input('Stiff.txt', x)

d1EI[0] = factor2*(-3*ysref[0] + 4*ysref[1] - ysref[2])/(2*h)+(1-factor2)*br*EIy[0]
d1EI[1] = factor2*( ysref[2] - ysref[0] )/(2*h)+(1-factor2)*br*EIy[1]
d1EI[n-1] = factor2*(3*ysref[n-1] - 4*ysref[n-2] + ysref[n-3])/(2*h)+\
        (1-factor2)*br*EIy[n-1]
d1EI[n-2] = factor2*( ysref[n-1] - ysref[n-3] )/(2*h)+0.85*br*EIy[n-2]
for i in range(2, n-2):
    d1EI[i] = factor2*(-ysref[i+2] + 8*ysref[i+1] - 8*ysref[i-1] +\
                    ysref[i-2])/(8*h)+(1-factor2)*br*EIy[i]

d2EI[0] = factor2*(2*ysref[0] - 5*ysref[1] + 4*ysref[2] - ysref[3])/(h**2)+ \
    (1-factor2)*(br**2)*EIy[0]
d2EI[n-1] = factor2*(2*ysref[n-1]-5*ysref[n-2]+4*ysref[n-3]-ysref[n-4])/(h**2)+\
    (1-factor2)*(br**2)*EIy[0]
for i in range(1, n-1):
    d2EI[i] = factor2*(ysref[i+1] - 2*ysref[i] + ysref[i-1])/(h**2) +\
        (1-factor2)*(br**2)*EIy[0]

# VisualizaÃ§Ã£o das condiÃ§Ãµes iniciais

fig = plt.figure(figsize=(21,13))
plt.box(False)
plt.xticks([])
plt.yticks([])

plt.subplot(3, 2, 1)
plt.plot(x, mc, color='g', linestyle='solid', label = "Modelo 1")
plt.plot(x, m, color='r', linestyle='solid', label = "Modelo 2 e 3")
plt.plot(x, m, color='k', linestyle='dashed', label = "DTU 10MW")
plt.grid(True)
plt.xlabel('z [m]')
plt.ylabel('Massa [kg/m]')
plt.legend() 

plt.subplot(3, 2, 2)
plt.plot(x, EIyc, color='g', linestyle='solid', label = "Modelo 1")
plt.plot(x, EIy, color='r', linestyle='solid', label = "Modelo 2")
plt.plot(x, EIy1, color='b', linestyle='solid', label = "Modelo 3")
plt.plot(x, ysref, color='k', linestyle='dashed', label = "DTU 10MW")
plt.grid(True)
plt.xlabel('z [m]')
plt.ylabel('Rigidez flapwise [kg.m2]')
plt.legend() 
#plt.xticks([])

plt.subplot(3, 2, 3)
plt.plot(x, bcc, color='g', linestyle='solid', label = "Modelo 1")
plt.plot(x, b, color='r', linestyle='solid', label = "Modelo 2 e 3")
plt.grid(True)
plt.xlabel('z [m]')
plt.ylabel('Amortecimento [kg/m.s]')
plt.legend() 

plt.subplot(3, 2, 4)
plt.plot(x, chordc, color='g', linestyle='solid', label = "Modelo 1")
plt.plot(x, chord, color='r', linestyle='solid', label = "Modelo 2 e 3")
plt.plot(x, chord, color='k', linestyle='dashed', label = "DTU 10MW")
plt.grid(True)
plt.xlabel('z [m]')
plt.ylabel('Corda [m]')
plt.legend() 

plt.subplot(3, 2, 5)
plt.plot(x, twist, color='r', linestyle='solid', label = "Modelo 2 e 3")
plt.plot(x, twist, color='k', linestyle='dashed', label = "DTU 10MW")
plt.grid(True)
plt.xlabel('z [m]')
plt.ylabel('Twist [rad]')
plt.legend() 

plt.subplot(3, 2, 6)
plt.plot((180/math.pi) * LeadAngle, Cl(LeadAngle, L, L), color='r', \
         linestyle='solid', label = 'Cl modelos 1, 2 e 3')
plt.plot(CLx, CLy, 'o', color='r', label = 'Cl  FFA-W3-241')  
plt.plot((180/math.pi) * LeadAngle, Cd(LeadAngle), color='b', \
         linestyle='solid', label = 'Cd modelos 1, 2 e 3')
plt.plot(CDx, CDy, 'o', color='b', label = 'Cd FFA-W3-241')
plt.grid(True)
plt.xlabel('Ângulo de ataque [ª]')
plt.ylabel('Coef []')
plt.legend()

plt.show()

for isamp in range(nsamp): 

    # InicializaÃ§Ã£o das variáveis
    
    t_S = []
    Theta_S = []
    U_S = []
    U_SM = []
    Sigma_S = []
    HW_S = []
    Alpha_S = []
    F_S = []
    M_S = []
    UW_S = []
    U0p_new = np.reshape(U0p_new, (len(U0p_new), 1))
    U0p_old = np.reshape(U0p_old, (len(U0p_old), 1))
    zf = []
    yf = []
    xf = []
    zf3 = []
    yf3 = []
    xf3 = []
    zf3_comp = []
    yf3_comp = []
    
    f1m = 1000 * np.ones(len(VelV))
    f1M = -1000 * np.ones(len(VelV))
    f2m = 1000 * np.ones(len(VelV))
    f2M = -1000 * np.ones(len(VelV))
    f3m = 1000 * np.ones(len(VelV))
    f3M = -1000 * np.ones(len(VelV))
    rf = np.zeros(len(VelV))
    
    r = 0
    
    for Vhub in range(Vmax, Vmin, -PV):
        
        r = r+1
    
        # Curva de acionamento do rotor    
    
        Rot[-r] = 0
        if Vhub <= Vco:
            Rot[-r] = Rot_max
            if Vhub <= Vr:
                Rot[-r] = TSR*Vhub/L 
                if Vhub < Vci:
                    Rot[-r] = 0
        if Rot[-r] > Rot_max:
            Rot[-r] = Rot_max
        
        if Rot[-r] > 0:
            pa = 1000
        else:
            pa = nt
        
        print("Sample = ", isamp , " VelV = ", Vhub, " Rot = ", \
              "{:.2f}".format(Rot[-r]))
        
        # Movimento flapwise (x)
        
        for j in range(0, nt):       
        
            # Controle de pitch
            
            # Define o setpoint com base na curvatura estática
            
            if j == 1:
                
                # Calcula a curvatura estática
                
                for p in range (0, n-1):
                    Q_ref[p] = CalcQ(Rho, chord[p], Vhub, Rot[-r], x[p] + offset, m[p], g, \
                                     CalcTheta(Rot[-r], Vhub, inc, x[p], twist[p]), \
                                     Cone, Phy[j], Tilt, h)
                        
                U_ref = np.reshape(U_ref, (len(U_ref), 1))
                Q_ref = np.reshape(Q_ref, (len(Q_ref), 1))
            
                U_ref = np.linalg.solve(K, Q_ref)
                
                Sigma_ref = np.array((2*U_ref[0] - 5*U_ref[1] +\
                            4*U_ref[2] - U_ref[3]) /(h**2))
                    
                Sigma_old = Sigma_ref
            
            if t[j] > (Tf - ta): 
                
                if j % nc == 0:
                    
                    # Calcula o erro
                    
                    err =  Sigma_ref - Sigma[j-1] # Termo proporcional  
                    
                    err_dot = (Sigma_old - Sigma[j-1])/(nc * k) # Termo derivativo
                    Sigma_old = Sigma[j-1]
                    
                    alpha_set = Kp * err + Kd * err_dot
                    
                    # Compatibilidade do sinal de referência
                    
                    if alpha_set > alpha_max:
                        alpha_set = alpha_max
                        
                    if alpha_set < alpha_min:
                        alpha_set = alpha_min
                        
                    alpha_rate_set = ( alpha_set - alpha[j-1] ) / (nc * k)
                    
                    # Inercia do atuador do pitch
                    
                    if alpha_rate_set > alpha_rate_max:
                        alpha_rate_set = alpha_rate_max
                        
                    if alpha_rate_set < alpha_rate_min:
                        alpha_rate_set = alpha_rate_min
                        
                    d_alpha = alpha_rate_set * k
                
                # Atualiza pitch
            
                alpha[j] = alpha[j-1] + d_alpha
                
                if alpha[j] > alpha_max:
                    alpha[j] = alpha_max
                    
                if alpha[j] < alpha_min:
                    alpha[j] = alpha_min
        
            # DeterminaÃ§Ã£o da turbulência
            
            if (j % int(0.083/k)) == 0:
                
                Turb = NormalDist(0, DesV).inv_cdf(random.random())
        
            # ConstruÃ§Ã£o da matriz de rigidez
            
            if (j % pa) == 0:
            
            # CondiÃ§Ãµes de contorno em x = 0
            
                #1 - Desloc zero na raÃ­z (x=0)
            
                l[0] = 1 * (E * Iy) 
                K = [l]
                l    = [0]*n
                
                    #2 - InclinaÃ§Ã£o zero na raÃ­z (x=0)
                
                A  = CalcA(EIy[1], h)           
                B  = CalcB(m[1], g, L, x[1], Phy[j], Tilt, Cone, Rot[-r], h, \
                           d2EI[1]/factor)                      
                C  = CalcC(m[1], Rot[-r], x[1], g, Phy[j], Tilt, Cone, h)
                D = CalcD(d1EI[1]/factor, h)
                
                l[0] = -4*A +16*B -8*C + 2*D
                l[1] = 7*A -31*B + C - D
                l[2] = -4*A +16*B +8*C - 2*D
                l[3] = A - B - C + D
                
                K = np.r_[K, [np.array(l)]]
                l = [0]*n
            
            Uhub[j] = Uw[0] = Vhub + Turb
            Q[0] = 0
            
            for i in range(0, n-4):
                
                Uw[i+1] = Vhub*((Hhub + x[i+2]*math.cos(Phy[j]))/Hhub)**CisCoeff +\
                            Turb
                
                # Correção da inclinação da pá
                
                inc = 0.95*math.atan((U0p_old[i] - U0p_old[i+2])/(2*h))
                
                Theta = CalcTheta(Rot[-r], Uw[i+1], inc, x[i+1] + offset, twist[i+1]) - alpha[j]    
            
                Q[i+1]  = CalcQ(Rho, chord[i+1], Uw[i+1], Rot[-r], x[i+1] + offset,\
                        m[i+1], g, Theta, Cone, Phy[j], Tilt, h)
                
                # DeterminaÃ§Ã£o da rigidez
                
                if (j % pa) == 0:
                    
                    A  = CalcA(EIy[i+2], h)          
                    B  = CalcB(m[i+2], g, L, x[i+2], Phy[j], Tilt, Cone, Rot[-r], h,\
                               d2EI[i+2]/factor)           
                    C  = CalcC(m[i+2], Rot[-r], x[i+2], g, Phy[j], Tilt, Cone, h)
                    D = CalcD(d1EI[i+2]/factor, h)
                    # Diff finitas
                    
                    l[i]   = A - B + C - D
                    l[i+1] = -4*A +16*B -8*C + 2*D
                    l[i+2] = 6*A -30*B
                    l[i+3] = -4*A +16*B +8*C - 2*D
                    l[i+4] = A - B - C + D
                    
                    K = np.r_[K, [np.array(l)]]
                    l = [0]*n
            
            # CondiÃ§Ãµes de contorno em x = L
            
                #3 - Cisalhamento nulo na extremidade da pÃ¡ (x=L)
            
            if (j % pa) == 0:    
            
                A  = CalcA(EIy[-2], h)           
                B  = CalcB(m[-2], g, L, x[-2], Phy[j], Tilt, Cone, Rot[-r], h, \
                           d2EI[-2]/factor)                
                C  = CalcC(m[-2], Rot[-r], x[-2], g, Phy[j], Tilt, Cone, h)
                D = CalcD(d1EI[-2]/factor, h)
                    
                l[-4] = A - B + C -D
                l[-3] = -4*A +16*B -8*C +2*D
                l[-2] = 6*A -30*B -1*(A - B - C + D)
                l[-1] = -4*A +16*B +8*C +2*(A - B - C + D) - 2*D
                K = np.r_[K, [np.array(l)]]
                l = [0]*n
            
            Uw[-2] = Vhub*((Hhub + x[-2] * math.cos(Phy[j]))/Hhub)**CisCoeff + Turb
            
            Theta = CalcTheta(Rot[-r], Uw[-2], inc, x[-2] + offset, twist[-2]) - alpha[j]
        
            Q[-2] = CalcQ(Rho, chord[-2], Uw[-2], Rot[-r], x[-2] + offset, m[-2], g,\
                            Theta, Cone, Phy[j], Tilt, h)  
            
                #4 - Momento fletor nulo na extremidade da pÃ¡ (x=L)
            
            if (j % pa) == 0:
                
                A  = CalcA(EIy[-1], h)               
                B  = CalcB(m[-1], g, L, x[-1], Phy[j], Tilt, Cone, Rot[-r], h, \
                           d2EI[-2]/factor)  
                C  = CalcC(m[-1], Rot[-r], x[-1], g, Phy[j], Tilt, Cone, h)
                D = CalcD(d1EI[-1]/factor, h)
                    
                l[-3] = A - B + C - D + (A - B - C + D)
                l[-2] = -4*A +16*B -8*C +2*D -1*(-4*A +16*B +8*C -2*D) -4*(A -B -C +D)
                l[-1] = 6*A -30*B +2*(-4*A +16*B +8*C -2*D) +4*(A - B - C +D)
                K = np.r_[K, [np.array(l)]]
            
            Uw[-1] = Vhub*((Hhub + x[-1] * math.cos(Phy[j]))/Hhub)**CisCoeff + Turb
            
            Theta = CalcTheta(Rot[-r], Uw[-1], inc, x[-1] + offset, twist[-1]) - alpha[j]
        
            Q[-1] = CalcQ(Rho, chord[-1], Uw[-1], Rot[-r], x[-1] + offset, m[-1], g,\
                            Theta, Cone, Phy[j], Tilt, h) 
            
            # AdequaÃ§Ã£o do sistema        
            
            if (j % pa) == 0:
                
                K = np.delete(K, 0, 1)
                K = np.delete(K, 0, 0)
            
            Q[1] = 0
            Q = np.reshape(Q, (len(Q), 1))
            
            # Freqs naturais
            
            eigvals, eigvecs = la.eig(np.dot(la.inv(Ms),K))
            f1 = math.sqrt(eigvals[n-2].real)/(2*math.pi)
            f2 = math.sqrt(eigvals[n-3].real)/(2*math.pi)
            f3 = math.sqrt(eigvals[n-4].real)/(2*math.pi)
            fnn = [f1, f2, f3]
            fnn.sort()
            f1, f2, f3 = fnn[0], fnn[1], fnn[2]
            
            if f1 < f1m[r-1]:
                f1m[r-1] = f1
            if f1 > f1M[r-1]:
                f1M[r-1] = f1
                
            if f2 < f2m[r-1]:
                f2m[r-1] = f2
            if f2 > f2M[r-1]:
                f2M[r-1] = f2
                
            if f3 < f3m[r-1]:
                f3m[r-1] = f3
            if f3 > f3M[r-1]:
                f3M[r-1] = f3
            
            rf[r-1] = Rot[-r]/(2 * math.pi)
            
            # RepresentaÃ§Ã£o no espaÃ§o de estados
            
            a1 = np.zeros((n-1, n-1))
            a2 = np.identity(n-1)
            a3 = -1 * np.matmul( la.inv(Ms) , K )
            a4 = -1 * np.matmul( la.inv(Ms) , Bs )
            
            a5 = np.c_[a1, a2]
            a6 = np.c_[a3, a4]
            As = np.r_[a5, a6]
            
            F = np.r_[np.zeros((n-1, 1)), np.matmul( la.inv(Ms) , Q )]
            
            # IntegraÃ§Ã£o
            
            U0p_new = np.matmul( (np.identity(2*(n-1)) + k * As \
            + ((k**2)/2) * np.linalg.matrix_power(As, 2) \
            + ((k**3)/6) * np.linalg.matrix_power(As, 3) \
            + ((k**4)/24) * np.linalg.matrix_power(As, 4)) , U0p_old ) \
            + np.matmul((k * np.identity(2*(n-1)) + ( ( k**2 ) /2) * As \
            + ((k**3)/6) * np.linalg.matrix_power(As, 2) \
            + ((k**4)/24) * np.linalg.matrix_power(As, 3)) , F)
            
            # CÃ¡lculo da curvatura, empuxo e momento da raiz da pÃ¡
            
            Sigma[j] = (2*U0p_old[0] - 5*U0p_old[1] +\
                        4*U0p_old[2] - U0p_old[3]) /(h**2)
            
            x_moment = np.delete(x, 0, 0)
            for index in range(0, n-1):
                f[index] = Q[index] * h
                    
            Empuxo[j] = np.sum(f) / 1000
            Momento[j] = np.dot(x_moment, f) / 1000
                
            # AtualizaÃ§Ã£o dos vetores deslocamento
            
            U0p_old = U0p_new
            
            U0p_new = np.reshape(U0p_new, (len(U0p_new), 1))
            U0p_old = np.reshape(U0p_old, (len(U0p_old), 1))
            
            # AtualizaÃ§Ã£o da posiÃ§Ã£o angular da pÃ¡
            
            if j < nt - 1:   
                Phy[j+1] = Phy[j] + Rot[-r] * k
            
            # Registrando vetores para anÃ¡lise e animaÃ§Ã£o
            
            if (j % rate == 0 ):
                
                U_S1 = reduce(concat, U0p_new.tolist())
                U_S.append(U_S1)
                
                if Vhub == Vmax:
    
                    t_S.append(t[j])
                    Theta_S.append(Phy[j] % (2*math.pi))
                    U_S1M = reduce(concat, U0p_new.tolist())
                    U_SM.append(U_S1M)
                    Sigma_S.append(Sigma[j]) 
                    HW_S.append(Uhub[j])
                    Alpha_S.append(alpha[j])
                    F_S.append(Empuxo[j])
                    M_S.append(Momento[j])
                    UW_S.append(Uw[:])
                
            if t[j] > (Tf - ta) and j % nc == 0:
                
                # Registrando os extremos
                
                if U_S1[n-2] < Max_U[r-1][isamp]:
                    Max_U[r-1][isamp] = U_S1[n-2]
                if Empuxo[j] < Max_Fx[r-1][isamp]:
                    Max_Fx[r-1][isamp] = Empuxo[j]
                if Momento[j] < Max_My[r-1][isamp]:
                    Max_My[r-1][isamp] = Momento[j]  
                if Sigma[j] < Max_Sigma[r-1][isamp]:
                    Max_Sigma[r-1][isamp] = Sigma[j]
                if np.abs(alpha[j]) > Max_Alpha[r-1][isamp]:
                    Max_Alpha[r-1][isamp] = np.abs(alpha[j])
                    
            if j == int(5*math.pi/(2*Rot[-r]*k)):
                
                Q_comp[-r] = np.array(Q[:])                     
        
        Sigma_rms[r-1][isamp] = np.sqrt(np.mean((Sigma[-int(ta/k):] - Sigma_ref)**2))
        Sigma_n1[r-1][isamp] = np.mean(np.abs(Sigma[-int(ta/k):] - Sigma_ref))
        Med_U[r-1][isamp] = (sum(element[n-2] for element in U_S))/len(U_S)
        
        if Vhub == 11:       
            
            fig = plt.figure(figsize=(13,8))
            
            plt.subplot(5, 1, 1)
            plt.title("Efeito do vento na pa")
            plt.plot(t, Uhub, color='r', linestyle='solid', linewidth=0.5)
            plt.ylabel('Vento no Hub [m/s]')
            plt.grid(True)
            
            plt.subplot(5, 1, 2)
            plt.plot(t, Empuxo, color='r', linestyle='solid')
            plt.ylabel('Empuxo [kN]')
            plt.grid(True)
            plt.xlabel('t [s]')
            
            plt.subplot(5, 1, 3)
            plt.plot(t, Momento, color='r', linestyle='solid')
            plt.ylabel('Momento [kNm]')
            plt.grid(True)
            plt.xlabel('t [s]')
                    
            plt.subplot(5, 1, 4)
            plt.plot(t, Sigma, color='r', linestyle='solid')
            plt.ylabel('Curvatura [1/m]')
            plt.grid(True)
            plt.xlabel('t [s]')
            
            plt.subplot(5, 1, 5)
            plt.plot(t, alpha, color='r', linestyle='solid')
            plt.ylabel('Pitch [rad]')
            plt.grid(True)
            plt.xlabel('t [s]')
            
            plt.show() 
            
            # Creating the polar scatter plot
            
            plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, polar=True)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_ylim(0,0.0025)
            ax.set_yticks(np.arange(0,0.0025,0.0005))
            ax.scatter(Phy[-int(ta/k):], np.abs(Sigma[-int(ta/k):] - Sigma_ref), \
                       s = 1, color='k', marker='+', cmap = mycmap, alpha = 0.7)
            circle = pl.Circle((0, 0), Sigma_rms[0][isamp], transform=ax.transData._b, \
                        color = "r", linestyle = 'solid', linewidth = 10, fill = False)
            ax.add_artist(circle)
            plt.title('Erro absoluto da curvatura x Azimute', fontsize=15)
            plt.show()
            
            plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, polar=True)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_ylim(0,alpha_max*180/math.pi)
            ax.set_yticks(np.arange(0,alpha_max*180/math.pi,0.5))
            ax.scatter(Phy[-int(ta/k):], np.abs(alpha[-int(ta/k):])*180/math.pi, \
                       s = 1, color='k', marker='+', cmap = mycmap, alpha = 0.7)
            plt.title('Alpha x Azimute', fontsize=15)
            plt.show()
    
        # Análise cascata da FFT do sinal
        
        window = np.blackman(int(ta/k))
        
        xs = np.linspace(0.0, 1.0/(2.0*k), int(ta/k)//2)
        ys = (scipy.fftpack.fft(window * Sigma[-int(ta/k):]))
        ys = 2.0/int(ta/k) * np.abs(ys[:int(ta/k)//2])
        
        # Filtro Savitzky-Golay
        
        xsf = savgol_filter(xs, 8, 3)
        ysf = savgol_filter(ys, 8, 3)
     
        xf.append(xsf)
        yf.append(Vhub)
        zf.append(2.0/int(ta/k) * np.abs(ysf[:int(ta/k)//2]))
        
        # Criando degrade para suavizar efeito pixel
        
        new_xsf = np.linspace(min(xsf), max(xsf), num=5*len(xsf))
        new_ysf = np.interp(new_xsf, xsf, ysf)
        
        xf3.append(new_xsf)
        yf3.append(Vhub)
        zf3.append(2.0/int(ta/k) * np.abs(new_ysf[:int(5*ta/k)//2]))
    
    # Carregamento na pá
    
    #fig = plt.figure(figsize=(13,8))
    #plt.title("Carregamento ao longo da pá")
    #plt.plot(x, Q_ref2[-1], color=Lcolor[-1], linestyle='dashed', linewidth=1, \
    #         label = "DTU 10MW")
    #plt.plot(x[1:], -1*Q_comp[-1], 'o', color=Lcolor[-1], label = "Modelo 1")
    #if Vmin > 5 and Vmax > 10:
    #    for j in range( 2, 6):
    #        plt.plot(x, Q_ref2[-j], color=Lcolor[-j], linestyle='dashed', linewidth=1)
    #        plt.plot(x[1:], -1*Q_comp[-j], 'o', color=Lcolor[-j])
    #plt.ylabel('Vento no Hub [m/s]')
    #plt.grid(True)
    #plt.legend()
    
    #plt.show() 
    
    # Heat map
    
    for j in range( 0, 5):
        for i in range( 0, len(yf3)-1):
            zf3.insert( 2*i + 1, (0.5 * np.add(np.array(zf3[2*i]), \
                                               np.array(zf3[2*i+1]))))
            yf3.insert( 2*i + 1, (0.5 * np.add(np.array(yf3[2*i]), \
                                               np.array(yf3[2*i+1]))))
    
    yf2 = yf
    while rf[-1] == 0:
        rf = np.delete(rf, -1)
        yf2 = np.delete(yf2, -1)
    
    while rf[0] == 0:
        rf = np.delete(rf, 0)
        yf2 = np.delete(yf2, 0)
    
    z_min, z_max = 0.0, np.abs(zf3).max()/2
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(f1m, yf, color = 'c', linestyle='dashed', linewidth=1)
    ax.plot(f1M, yf, color = 'c', linestyle='dashed', label = '1a fn', linewidth=1)
    ax.plot(f2m, yf, color = 'c', linestyle='dashdot', linewidth=1)
    ax.plot(f2M, yf, color = 'c', linestyle='dashdot', linewidth=1, label ='2a fn')
    ax.plot(f3m, yf, color = 'c', linestyle='dotted', linewidth=1)
    ax.plot(f3M, yf, color = 'c', linestyle='dotted', linewidth=1, label = '3a fn')
    ax.plot(rf, yf2, color = 'c', linestyle='-', linewidth=1, label = 'Rotação')
    ax.set_xscale('log')
    ax.set( xlim=(0.05, 10) )
    cmap = ax.pcolormesh(new_xsf, yf3, zf3, cmap = mycmap, alpha = 0.7, \
                         vmin = z_min, vmax = 2.5e-9)#z_max/8)

    fig.colorbar(cmap)
    plt.ylabel('Vhub [m/s]')
    plt.xlabel('Freq [Hz]')
    plt.title("Heatmap da FFT da curvatura na raiz da pá")
    plt.legend(loc='lower right')
    plt.show(fig)
    
    # 3D surface
    
    for i in range(len(xs)):
        if xs[i] > 0.05:
            xmin = i
            break
    
    for i in range(len(xs)):
        if xs[i] > 10:
            xmax = i
            break
    
    xm, ym = np.meshgrid(xs[xmin:xmax], yf)
    xm = np.log10(xm)
    zm = np.vstack(zf)
    zm = zm[:,xmin:xmax]
    
    fig = plt.figure(figsize =(14, 9))
    #ax = Axes3D(fig)
    ax = plt.axes(projection ='3d')
    
    for i in range(len(yf)):
        if (i % 2 == 0):
            ax.plot3D(xm[i,:], ym[i,:], zm[i,:], color = Lcolor[int(i%7)])
    #ax.scatter(xm, ym, zm, c=zm, cmap='viridis', linewidth=0.5);
     
    # Adding labels
    ax.set_xlabel('log(Freq)')
    ax.set_xlim(-1.3, 1.0)
    #ax.set_xscale('log')
    ax.set_ylabel('Vhub [m/s]')
    ax.set_ylim( Vmin, Vmax + 1)
    ax.set_zlabel('Amplitude da curvatura [1/m]')
    ax.set_zlim(np.min(zm), 2.5e-9)
    ax.set_title('Análise da curvatura na rai­z da pá')
    
    plt.show(fig)

# Curva de acionamento do rotor

fig = plt.figure(figsize=(8,5))

plt.plot(VelV, Rot, color='k', linestyle='dashed', label = 'Sol Numérica')

plt.grid(True)
plt.xlabel('Veloc. Hub (m/s)')
plt.ylabel('Rot (rad/s)')
plt.legend() 
plt.title("Curva de acionamento do rotor")
plt.show()

# Estatísticas

print("Number of samples / wind = ", nsamp)
print('Máximo Empuxo: [kN]')
PosTable(Max_Fx, Vmax, Vmin, PV)
print('Máximo Momento: [kNm]')
PosTable(Max_My, Vmax, Vmin, PV)
print('Máxima Deflexão: [m]')
PosTable(Max_U, Vmax, Vmin, PV)
print('Deflexão Média: [m]')
PosTable(Med_U, Vmax, Vmin, PV)
print('Erro Curvatura RMS: [1000/m]')
Sigma_rms = 1000 * np.array(Sigma_rms)
PosTable(Sigma_rms, Vmax, Vmin, PV)
print('Erro Curvatura Norma 1: [1000/m]')
Sigma_n1 = 1000 * np.array(Sigma_n1)
PosTable(Sigma_n1, Vmax, Vmin, PV)
print('Máximo Pitch: [º]')
Max_Alpha = (180 / math.pi) * np.array(Max_Alpha)
PosTable(Max_Alpha, Vmax, Vmin, PV)

print("Run time: %.1f seconds" % (time.time() - start_time))

# AnimaÃ§Ã£o dos resultados

pygame.init()
step = 0
size = (640, 480)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
surface = pygame.display.set_mode(size)
pygame.display.set_caption("Wind Blade Simulator")
clock = pygame.time.Clock()
fps = 1/(rate * k)
run = True
while run:
    if pygame.event.peek(pygame.QUIT):
        break
    clock.tick(fps)
    surface.fill(WHITE)

    # PÃ¡s girando no cÃ­rculo

    pygame.draw.circle(surface, BLACK, (150, 200), L+2)
    pygame.draw.circle(surface, WHITE, (150, 200), L)
    
    pygame.draw.line(surface, BLACK, (145, 200), (135, 350), 2)
    pygame.draw.line(surface, BLACK, (155, 200), (165, 350), 2)
    pygame.draw.line(surface, BLACK, (135, 350), (165, 350), 2)

    pygame.draw.line(surface, RED, (150, 200),(150 - L*math.sin(-Theta_S[step]),\
                   (200 - L*math.cos(-Theta_S[step]))), 8)
    
    pygame.draw.line(surface, RED, (150, 200), (150 - L*math.sin(-Theta_S[step]\
    + 2*math.pi/3), (200 - L*math.cos(-Theta_S[step] + 2*math.pi/3))), 8)
        
    pygame.draw.line(surface, WHITE, (150, 200), (150 - L*math.sin(-Theta_S[step]\
    + 2*math.pi/3), (200 - L*math.cos(-Theta_S[step] + 2*math.pi/3))), 4)

    pygame.draw.line(surface, RED, (150, 200), (150 - L*math.sin(-Theta_S[step]\
    - 2*math.pi/3), (200 - L*math.cos(-Theta_S[step] - 2*math.pi/3))), 8)
  
    pygame.draw.line(surface, WHITE, (150, 200), (150 - L*math.sin(-Theta_S[step]\
    - 2*math.pi/3), (200 - L*math.cos(-Theta_S[step] - 2*math.pi/3))), 4)
        
    pygame.draw.circle(surface, BLACK, (150, 200), 10)
    pygame.draw.circle(surface, BLACK, (345, 200), 10)

    # Extremos
    
    pygame.draw.line(surface, RED, (350, 200 + 5*U_max), \
                     (350 + 2*L, 200 + 5*U_max), 2)
        
    pygame.draw.line(surface, YELLOW, (350, 200 + 5*12.4), \
                         (350 + 2*L, 200 + 5*12.4), 2)

    # Deslocamentos da pÃ¡

    for i in range(0, n-2):
        pygame.draw.line(surface, BLACK, (350 + 2*x[i]+h, 200 - \
                  5*U_SM[step][i]), (350 + 2*x[i+1]+h, 200 - 5*U_SM[step][i+1]), 1)
        pygame.draw.circle(surface, RED, (350 + 2*x[i]+h, 200 -5*U_SM[step][i]),2)
        pygame.draw.line(surface, GREEN, (350 + 2*x[i]+h, 200), \
                         (350 + 2*x[i]+h, 200 - 5*UW_S[step][i+1]), 1)
    pygame.draw.circle(surface, RED, (350 + 2*x[n-2]+h, 200 - \
                                      5*U_SM[step][n-2]),2)
    pygame.draw.line(surface, GREEN, (350 + 2*x[n-2]+h, 200), \
                     (350 + 2*x[n-2]+h, 200 - 5*UW_S[step][n-2]), 1)
        
    # Pitch
    
    pygame.draw.line(surface, RED, (550, 350), \
                     (550 - 0.5*L*math.cos(5*Alpha_S[step]), \
                     350 - 0.5*L*math.sin(5*Alpha_S[step])), 5)
    pygame.draw.line(surface, RED, (550, 350), \
                         (550 + 0.5*L*math.cos(5*Alpha_S[step]), \
                         350 + 0.5*L*math.sin(5*Alpha_S[step])), 5)
        
    # Medidas

    font = pygame.font.SysFont('Calibri', 12, True, False)
    
    text1 = font.render('Time [s]', True, BLACK)
    surface.blit(text1, [50, 50])
    text1a = font.render(str("{:.1f}".format(t_S[step])), True, BLACK)
    surface.blit(text1a, [150, 50])
    
    text2 = font.render('HubWind [m/s]', True, BLACK)
    surface.blit(text2, [350, 50])
    text2a = font.render(str("{:.1f}".format(HW_S[step])),\
                         True, BLACK)
    surface.blit(text2a, [500, 50])
    
    text3 = font.render('Azimute [º]', True, BLACK)
    surface.blit(text3, [50, 75])
    text3a = font.render(str("{:.1f}".format(math.degrees(Theta_S[step]))),\
                         True, BLACK)
    surface.blit(text3a, [150, 75])
    
    text4 = font.render('Pitch [º]', True, BLACK)
    surface.blit(text4, [200, 350])
    text4a = font.render(str("{:.2f}".format(math.degrees(Alpha_S[step]))),\
                         True, BLACK)
    surface.blit(text4a, [350, 350])

    text5 = font.render('Curvatura [1000/m]', True, BLACK)
    surface.blit(text5, [50, 400])
    text5b = font.render(str("{:.1f}".format(Sigma_S[step]*10**3)), True, \
                         BLACK)
    surface.blit(text5b, [50, 425])
    
    text6 = font.render('Empuxo [kN]', True, BLACK)
    surface.blit(text6, [200, 400])
    text6b = font.render(str("{:.1f}".format(F_S[step])), True, BLACK)
    surface.blit(text6b, [200, 425])
    
    text7 = font.render('Momento [kNm]', True, BLACK)
    surface.blit(text7, [350, 400])
    text7b = font.render(str("{:.1f}".format(M_S[step])), True, BLACK)
    surface.blit(text7b, [350, 425])
    
    text8 = font.render('Desloc. ponta [m]', True, BLACK)
    surface.blit(text8, [500, 400])
    text8b = font.render(str("{:.1f}".format(U_SM[step][n-2])), True, BLACK)
    surface.blit(text8b, [500, 425])
    
    text9 = font.render('Powered by LukinTechWorks(c)copyright2024', \
                        True, BLACK)
    surface.blit(text9, [400, 460])
    
    pygame.display.flip()
    
    step += 1

    if (step > len(t_S) - 1):
        step = 0

pygame.quit()