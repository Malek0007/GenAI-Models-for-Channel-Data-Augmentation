# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:36:45 2021

@author: RZ268260
---------------- Uniform  pilot assignement/simulation over different users selected randomly -----------------------
---------------- SINR theo and SINR sim / SEtheo and SE sim -------------------------------
PZF with PAPR Reduction
"""

import numpy as np
import SignalTools
import matplotlib.pyplot as plt
import json
#import tensorflow as tf
# Function definition
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump
def gram_schmidt(M):
    n = M.shape[1]
    for j in range(n):
        for k in range(j):
            M[:,j] -= np.dot(M[:,k], M[:,j])*M[:,k]
        M[:,j] = M[:,j]/np.linalg.norm(M[:,j])
    return M
def randU(n):
    X = np.random.randn(n,n) + 1j*np.random.randn(n,n)
    Q, R = np.linalg.qr(X)
    R = np.diag(np.diag(R)/np.abs(np.diag(R)))
    return Q.dot(R)
def limiter(vin,amin,amax):
    a0 = np.abs(vin)
    theta = np.angle(vin)
    ac = np.clip(a0,amin,amax)
    vout = ac * np.exp(1j * (theta))
    return vout
def hpa_sspa_modif_rapp(vin, Vsat, p, q, G, A, B):
    a0 = np.abs(vin)
    theta = np.angle(vin)
    Am = (G * a0) / ((1 + (G * a0 / Vsat) ** (2 * p)) ** (1 / (2 * p)))
    Bm = (A * (a0 ** q)) / ((1 + (a0 / B) ** (q)))
    vout = Am * np.exp(1j * (theta + Bm))
    return vout
def find_K0_sigma2_d(IBO):
    xin = (1 / np.sqrt(2)) * (np.random.randn(1, 10000) + 1j * np.random.randn(1, 10000))
    coeff_IBO_m1dB = (
        val_IBO_m1dB * np.sqrt((1 / np.var(xin))) * np.sqrt(10 ** (-IBO / 10))
    )
    vin = coeff_IBO_m1dB * xin
    vout = hpa_sspa_modif_rapp(vin, Vsat, p, q, G, A, B)    
    K0 = np.mean(vout * np.conj(vin)) / np.mean(np.absolute(vin) ** 2)
    sigma2_d = np.mean(np.abs(vout - K0 * vin)**2)
    return (K0, sigma2_d)

def design_precoder_local_PFZF(H3barl,Thetalk):
    W_PFZFl = np.zeros((M,M-1,L,Mfft),dtype=complex)
    W_MRTl = np.zeros((M,Tau_p,L,Mfft),dtype=complex)
    Wlk = np.zeros((M,Tau_p,L,Mfft),dtype=complex)
    for l in range(0,L):
        for mf in range(0,Mfft):
            W_PFZFl[:,:Tau_ss1[l],l,mf] =  H3barl[:,:,l,mf].dot(ESl[l,:,:Tau_ss1[l]]).dot(np.linalg.inv(np.transpose(np.conjugate(ESl[l,:,:Tau_ss1[l]])).dot(np.transpose(np.conjugate(H3barl[:,:,l,mf]))).dot(H3barl[:,:,l,mf]).dot(ESl[l,:,:Tau_ss1[l]]))) #W_FZFl[:,:,l,mf] = (np.linalg.inv((H3barl[:,:,l,mf]).dot(np.transpose(np.conjugate(H3barl[:,:,l,mf]))) + 0*SIGl + Pl)).dot(H3barl[:,:,l,mf]) # 
            W_MRTl[:,:,l,mf] =  H3barl[:,:,l,mf]
    for l in range(0,L):
        for mf in range(0,Mfft):
            for k in range(0,Tau_ss1[l]):
                ii = int(Ilk[l,k])
                Wlk[:,ii,l,mf] =  np.sqrt((M-Tau_ss1[l])*Thetalk[l,ii])*W_PFZFl[:,k,l,mf] #W_FZFl[:,k,l,mf]/np.real(K*np.transpose(np.conjugate(H3barl[:,k,l,mf])).dot(W_FZFl[:,k,l,mf])) #
            for k in range(Tau_ss1[l],Tau_p):
                ii = int(Ilk[l,k])
                Wlk[:,ii,l,mf] =  W_MRTl[:,ii,l,mf]/np.sqrt(M*Thetalk[l,ii])
    
    WW = np.zeros((M,(int(K/Tau_p)+1)*Tau_p,L,Mfft),dtype=complex)
    for tp in range(0,int(K/Tau_p)+1):
        WW[:,tp*Tau_p:(tp+1)*Tau_p,:,:] = Wlk

    W_PZFlk = WW[:,:K,:]                
    Wlk = W_PZFlk
    return Wlk

def papr_reduction(XX,V00,Niter, Tau_ssl):
    Xout = XX

    XDD = np.zeros((M,Mfft),dtype=complex)

    for iNiter in range(0,Niter):
        disXT = np.zeros((M,Mfft),dtype=complex)
        xlOFDM_clip = np.zeros((M,Mfft),dtype=complex)
        xlOFDMpp = np.fft.ifft(Xout)
        for m in range(0,M):
            amax = 1.37*np.sqrt(np.mean(np.abs(xlOFDMpp[m,:]))**2)#np.sqrt(((M/(Tau_p - Tau_ssl))*((Mfft)/(Mfft-2*BG))))*np.sqrt(np.mean(np.abs(xlOFDMpp[m,:]))**2)
            xlOFDM_clip[m,:] = limiter(xlOFDMpp[m,:], 0, amax)#limiter(vin,amin,amax)#
            disXT[m,:] = np.reshape(xlOFDM_clip[m,:] - xlOFDMpp[m,:],(1,Mfft))
        # OFDm demodulation
        Xi =  np.fft.fft(disXT)
        for mf in range(0,Mfft):
            pchp = np.sum(np.abs(V00[:,:,BG].dot(Xi[:,BG]))*np.abs(Xi[:,BG]))/np.sum(np.abs(V00[:,:,BG].dot(Xi[:,BG])**2))
            Xout[:,mf] = XX[:,mf] + pchp*V00[:,:,mf].dot(Xi[:,mf])
            XDD[:,mf] = XDD[:,mf] + pchp*V00[:,:,mf].dot(Xi[:,mf])
        Xout[:,:BG] = np.zeros((M,BG),dtype=complex)
        Xout[:,Mfft-BG:] = np.zeros((M,BG),dtype=complex)
        XDD[:,:BG] = np.zeros((M,BG),dtype=complex)
        XDD[:,Mfft-BG:] = np.zeros((M,BG),dtype=complex)
    return XDD , Xout      
# System model
Mqam = 16
modulation_type = "qam"

PNiter = 0
BG = 1
Mfft = 64 + 2*BG

IBO = 3; 
name_ibo = IBO*10
IBOr= np.sqrt(10**(-IBO/10))
Nsymb=168
p = 100 #1.1
q = 4
Vsat = 1.9
G = 16 #16
A = 0 #-345
B = 0.17

val_IBO_m1dB = Vsat/G

K0, sigma2_d = find_K0_sigma2_d(IBO)

# User grouping related Precoding
nuTh1 = 1

# User grouping related PAPR reduction
nuTh2 = 1

N = 128  # taille du signal OFDM
L = 100
M = 16
K = 8
Tau_p = 8
Tau_c = 168
xi = 0.5

pin = (val_IBO_m1dB**2)/(10 ** (IBO / 10))
Pin = pin*M
Pinmw = 1000*Pin
clipping_threshold = 3.0  # Typiquement entre 2 et 4
    
sigma_sh_dB = 4
sigma_sh = 10**(sigma_sh_dB/10)
Kau = 0.5
APh = 10
UEh = 1.5

FcGHz = 3.5

wp_dBm = -93
wp = 0.001*(10**(wp_dBm/10))

Rho_l_max_dBm = 10*np.log10(Pinmw) - 0*wp_dBm
Rho_l_max = Pin # 0.001*(10**(Rho_l_max_dBm/10))

pk_dBm = 10*np.log10(100) - 0*wp_dBm
pk = 0.1 #0.001*(10**(pk_dBm/10))

G = 1
D = 1000

prec = "PZF"

Nsnap = 1
SEkTheo = np.zeros(Nsnap,dtype=float)
SEksim = np.zeros((K,Nsnap),dtype=float)
SEksim0 = np.zeros((K,Nsnap),dtype=float)
SE_individuel_vecteur = np.zeros((K,Nsnap),dtype=float)  # taille : (snapshots, utilisateurs)

for iNsnap in range(0,Nsnap):
    UOIindex = int(np.random.randint(0,K,1))
    SCOIindex = int(np.random.randint(BG,Mfft-BG,1))
    APs = np.zeros((L,3),dtype=float)
    APs[:,0] = list(frange(1,L+1,1))
    APs[:,1] = np.random.uniform(0,D,L)
    APs[:,2] = np.random.uniform(0,D,L)
    
    UEs = np.zeros((K,3),dtype=float)
    UEs[:,0] = list(frange(1,K+1,1))
    UEs[:,1] = np.random.uniform(0,D,K)
    UEs[:,2] = np.random.uniform(0,D,K)
    
    with open("distance_matrix.json", "r") as file:
        distance_matrix = np.array(json.load(file))  # Convertir en tableau NumPy

    # Calculer la transposée
    Dlk = distance_matrix.T  
    # Pilots
    Pilots = np.sqrt(Tau_p)*randU(Tau_p)
    # Uniform Pilot assignement
    userPilotIndex0 = list(frange(0,Tau_p,1))
    userPilotIndex00 = [] #10*np.ones((Tau_p*(int(K/Tau_p)+1)),dtype=int)
    for tp in range(0,int(K/Tau_p)+1):
        userPilotIndex00 = np.concatenate((userPilotIndex00,userPilotIndex0))
    userPilotIndex = userPilotIndex00[:K]
    
    # userPilotIndex0 = list(frange(0,Tau_p,1))
    # userPilotIndex1 = list(frange(0,K-Tau_p,1))
    # userPilotIndex = np.concatenate((userPilotIndex0,userPilotIndex1))
    PhiP = np.zeros((Tau_p,K),dtype=complex)
    for up in range(0,K):
        PhiP[:,up] = Pilots[:,int(userPilotIndex[up])]
    
    PL_lk_dB = -36.7*np.log10(Dlk) - 22.7 - 26*np.log10(FcGHz) #-30.5 - 36.7*np.log10(Dlk)
    PL_lk = 10**(PL_lk_dB/10)

    al0 = np.random.randn(L,1)
    al = np.repeat(al0,K,-1)
    bk0 = np.random.randn(K,1)
    bk00 = np.repeat(bk0,L,-1)
    bk = np.transpose(bk00)
    zlk = np.sqrt(Kau)*al + np.sqrt(1-Kau)*bk
    sigma_sh_Zlk = sigma_sh*zlk
    BETAlk = PL_lk*(10**(sigma_sh_Zlk/10))
    
    # SORT BETAlk
    Itp = np.eye(Tau_p)
    Ilk = np.zeros((L,Tau_p),dtype=int)
    
    BETAilk = np.zeros((L,Tau_p),dtype=float) 
    BETAmlk = np.zeros((L,Tau_p),dtype=float) 
    for l in range(0,L):
        for tp in range(0,Tau_p):
            BETAilk[l,tp] = max(BETAlk[l,tp:K:Tau_p]) 
            BETAmlk[l,tp] = min(BETAlk[l,tp:K:Tau_p])                
        tmp = np.argsort(BETAilk[l,:])
        Ilk[l,:] = tmp[::-1]
    
    BETAblk = np.zeros((L,Tau_p), dtype=float)
    Tau_ss1 = np.zeros((L), dtype=int)
    Tau_ss2 = np.zeros((L), dtype=int)
    TausMax = min(M-1,Tau_p)
    for l in range(0,L):
        BETAblk[l,:] = BETAilk[l,Ilk[l,:]]
        Tau_ss1[l] = TausMax
        Tau_ss2[l] = TausMax
        for ts in range(1,TausMax):
            if ((np.sum(BETAblk[l,:ts])/np.sum(BETAblk[l,:]))>nuTh1):
                Tau_ss1[l] = ts
                break
        for ts in range(1,TausMax):
            if ((np.sum(BETAblk[l,:ts])/np.sum(BETAblk[l,:]))>nuTh2):
                Tau_ss2[l] = ts
                break
    
    ESl = np.zeros((L,Tau_p,TausMax),dtype=int)        
    for l in range(0,L):
        for ts in range(0,Tau_ss1[l]):
            ii = int(Ilk[l,ts])
            ESl[l,:,ts] = Itp[:,ii]
    
    Pk = pk*np.diag(np.ones(K))
    
    # Channel estimation
    Clk = np.zeros((L,K),dtype=float)
    for k in range(0,K):
        UONIindex = [] # np.zeros((1,1),dtype=int)
        for kk in range(0,K):
            if (userPilotIndex[kk]==userPilotIndex[k]) & (kk!=k):
                UONIindex = np.concatenate((UONIindex,np.reshape(kk,(1))),0)
        for l in range(0,L):
            betalt = 0
            for ind in range(0,len(UONIindex)):
                betalt = betalt + BETAlk[l,int(UONIindex[ind])]
            Clk[l,k] = (np.sqrt(pk)*BETAlk[l,k])/((Tau_p*pk*(betalt+BETAlk[l,k])) + wp)
    
    NUlk = np.zeros((L,K),dtype=float)
    for k in range(0,K):
        UONIindex = [] # np.zeros((1,1),dtype=int)
        for kk in range(0,K):
            if (userPilotIndex[kk]==userPilotIndex[k]) & (kk!=k):
                UONIindex = np.concatenate((UONIindex,np.reshape(kk,(1))),0)
        for l in range(0,L):
            betalt = 0
            for ind in range(0,len(UONIindex)):
                betalt = betalt + BETAlk[l,int(UONIindex[ind])]
            NUlk[l,k] = ((pk*Tau_p)*(BETAlk[l,k]**2))/((pk*Tau_p*(betalt+BETAlk[l,k])) + wp)
            
    Thetalk = NUlk/(Clk**2)
            
    UONIindex = [] 
    for kk in range(0,K):
        if (userPilotIndex[kk]==userPilotIndex[UOIindex]) & (kk!=UOIindex):
            UONIindex = np.concatenate((UONIindex,np.reshape(kk,(1))),0)
            
    betalt = 0
    for ind in range(0,len(UONIindex)):
        betalt = betalt + BETAlk[l,int(UONIindex[ind])]
    
    
    RHOlk = np.zeros((L,K),dtype=float)
    for l in range(0,L):
        for k in range(0,K):
            RHOlk[l,k] = ((NUlk[l,k]/np.sum(NUlk[l,:]))*Rho_l_max)    # # Power control
    print("UOIindex",UOIindex)
    CPktheo0 = np.sqrt(RHOlk[:,UOIindex]*NUlk[:,UOIindex])
    CPktheo = (M-Tau_p)*(np.sum(CPktheo0))**2
    #print("CPktheo",CPktheo)
    
    PUktheo0 = 0
    for t in range(0,len(UONIindex)):
        PUktheo0 = PUktheo0 + (np.sum(np.sqrt(RHOlk[:,int(UONIindex[ind])]*NUlk[:,UOIindex])))**2
    PUktheo = (M-Tau_p)*PUktheo0
    #print("PUktheo",PUktheo)
    UIkttheo0=0
    for t in range(0,K):
        UIkttheo0 = UIkttheo0 + RHOlk[:,t]*(BETAlk[:,UOIindex]-NUlk[:,UOIindex])   
    UIkttheo = np.sum(UIkttheo0)
    #print("UIkttheo",UIkttheo)
    
    
    SINRktheo = CPktheo/(PUktheo + UIkttheo + wp)
    
    # # Dans la boucle :
    # for idx, kk in enumerate(range(k)):
    #     CPk = (M - Tau_p) * (np.sum(np.sqrt(RHOlk[:, int(kk)] * NUlk[:, int(kk)])))**2
        
    #     PUk = 0
    #     for j in range(0, K):
    #         if j != kk:
    #             PUk += (np.sum(np.sqrt(RHOlk[:, j] * NUlk[:, int(kk)])))**2
                
    #     UIk = np.sum(RHOlk[:, int(kk)] * (BETAlk[:, int(kk)] - NUlk[:, int(kk)]))

    #     SINRk = CPk / (PUk + UIk + wp)

    #     SE_individuel_vecteur[iNsnap, idx] = xi * (1 - (Tau_p / Tau_c)) * np.log2(1 + SINRk)
    # print("SE_individuel_vecteur",SE_individuel_vecteur)
    print("SINRktheo")
    print(SINRktheo)
    print("------------------------")
    SEkTheo[iNsnap] = xi*(1-(Tau_p/Tau_c))*np.log2(1+SINRktheo)
    SE_total = np.sum(SEkTheo)  # Somme de tous les éléments dans SEkTheo
    print("SE total :", SE_total)
    
    ##########################################################
    # Simulation
    # Channels
    Nreal = 25
    yk = np.zeros((K,Mfft,Nreal),dtype=complex)
    WHk = np.zeros((L,K,Nreal),dtype=complex)
    HDr = np.zeros((L,K,Nreal),dtype=complex)
    HDs = np.zeros((L,K,Nreal),dtype=complex)
    dr = np.zeros((K,Mfft,Nreal),dtype=complex)
    ds = np.zeros((K,Mfft,Nreal),dtype=complex)
    V0m = np.zeros((M,M,Nreal),dtype=complex)
    WHkt = np.zeros((L,K,Nreal,K),dtype=complex)
    # BITk = []
    Qk = np.zeros((Nreal),dtype=complex)
    for iNreal in range(0,Nreal):    
        H3 = (1 / np.sqrt(2)) * (np.random.randn(M, K, L, Mfft) + 1j * np.random.randn(M, K, L, Mfft))
        #In[1] Uplink Training phase
        realH3 = np.zeros((M,K,L,Mfft),dtype=complex)
        for l in range(0,L):
            for mf in range(0,Mfft):
                realH3[:,:,l,mf] = H3[:,:,l,mf].dot(np.diag(np.sqrt(BETAlk[l,:])))
        
        # channel in frequency domain
        realHFreq3 = realH3
        
        Yl = np.zeros((M,Tau_p,L,Mfft),dtype=complex)
        for l in range(0,L):
            for mf in range(0,Mfft):
                Nl = np.sqrt(wp/(2))*(np.random.randn(M,Tau_p) + 1j*np.random.randn(M,Tau_p))
                Yl[:,:,l,mf] = realHFreq3[:,:,l,mf].dot(np.sqrt(Pk).dot(np.conjugate(np.transpose(PhiP)))) + Nl 
        
        H3chp = np.zeros((M,K,L,Mfft),dtype=complex)
        for l in range(0,L):
            for k in range(0,K):
                for mf in range(0,Mfft):
                    H3chp[:,k,l,mf] =  Clk[l,k]*Yl[:,:,l,mf].dot(PhiP[:,k])
        
        H3barl = np.zeros((M,Tau_p,L,Mfft),dtype=complex)
        for l in range(0,L):
            for mf in range(0,Mfft):
                H3barl[:,:,l,mf] = Yl[:,:,l,mf].dot(Pilots)
        

        # # Downlink Data Transmission
        # PZF Precoding
        Wlk = design_precoder_local_PFZF(H3barl,Thetalk)
        
        #Channelm Null-space
        P = np.zeros((Tau_p,Tau_p,L),dtype=float)
        Bsotrted = np.zeros((L,Tau_p),dtype=float)
        for l in range(0,L):
            Bsotrted[l,:] = sorted(BETAlk[l,:Tau_p])
            for k in range(0,Tau_p):
                if (BETAilk[l,k]<Bsotrted[l,(Tau_p - Tau_ss2[l])]):
                    P[k,k,l] = 1/(RHOlk[l,k]) #100000000000000/RHOlk[l,k]#(1000000000000/RHOlk[l,k])

        H3barlV0 = np.zeros((M,Tau_p,L,Mfft),dtype=complex)
        for l in range(0,L):
            for k in range(0,Tau_p):
                H3barlV0[:,k,l,:] = H3barl[:,k,l,:]*(Thetalk[l,k]**2)
                
        V0 = np.zeros((M,M,L,Mfft),dtype=complex)
        for l in range(0,L):
            #P = 0*np.linalg.inv(np.diag(RHOlk[l,:Tau_p])) #♦0.00000003*(np.diag(RHOlk[l,:Tau_p])) #np.linalg.inv(np.diag(BETAlk[l,:Tau_p]))#np.linalg.inv(np.diag(BETAlk[l,:Tau_p]))#np.diag(RHOlk[l,:Tau_p]/np.min(RHOlk[l,:Tau_p]))#P = np.diag(RHOlk[l,:Tau_p]/np.mean(RHOlk[l,:Tau_p]))#0*np.diag(np.ones((K),dtype=float)) # 0.000001*np.linalg.inv(np.diag(BETAlk[l,:Tau_p])) # 0.0000000001*(np.diag(RHOlk[l,:Tau_p])) #À(10**-9)*np.linalg.inv(np.diag(BETAlk[l,:Tau_p])) # np.linalg.inv(np.diag(RHOlk[l,:Tau_p]))
            for mf in range(0,Mfft):
                V0[:,:,l,mf] = np.identity(M) - H3barlV0[:,:Tau_p,l,mf].dot(np.linalg.inv(np.transpose(np.conjugate(H3barlV0[:,:Tau_p,l,mf])).dot(H3barlV0[:,:Tau_p,l,mf]) + P[:,:,l]).dot(np.transpose(np.conjugate(H3barlV0[:,:Tau_p,l,mf]))))
        
        # Bit generation
        bitk = np.random.randint(2,size=int(np.log2(Mqam)*K*Mfft))
        qk = SignalTools.mapping(bitk, Mqam, modulation_type)
        qk = qk/np.sqrt(np.var(qk))
        qkP = np.reshape(qk,(K,Mfft)) 
        Qk[iNreal] = qkP[UOIindex,SCOIindex] #np.concatenate((Qk,qk[UOIindex]))
        
        Xl = np.zeros((M,L,Mfft),dtype=complex)
        xlOFDM = np.zeros((M,L,Mfft),dtype=complex)
        Xldd = np.zeros((M,L,Mfft),dtype=complex)

        for l in range(0,L):
            for mf in range(0,Mfft):
                Xl[:,l,mf] = Wlk[:,:,l,mf].dot(np.diag(np.sqrt(RHOlk[l,:]))).dot(qkP[:,mf])
            Xl[:,l,:BG] = np.zeros((M,BG),dtype=float)
            Xl[:,l,Mfft-BG:] = np.zeros((M,BG),dtype=float)
            # OFDM modulation
            xlOFDM[:,l,:] = np.fft.ifft(Xl[:,l,:])
        
        Xlp = np.zeros((M,L,Mfft),dtype=complex)
        XlD = np.zeros((M,L,Mfft),dtype=complex)
        Xdd = np.zeros((M,L,Mfft),dtype=complex)
        XRl = np.zeros((M,L,Mfft),dtype=complex)
        xlOFDMpr0 = np.zeros((M,L,Mfft),dtype=complex)
        xlOFDMpr = np.zeros((M,L,Mfft),dtype=complex)
        Coef_p = np.zeros((M,L),dtype=float)
        
        sigCompUps0 = np.zeros((M,L,Mfft),dtype=complex)
        for l in range(0,L):    
            for m in range(0,M):
                Coef_p[m,l] = 1#(np.sqrt(np.mean(np.abs(xlOFDM[m,l,:])**2)))
                xlOFDM[m,l,:] = xlOFDM[m,l,:]/Coef_p[m,l]
            # OFDm demodulation
            XRl[:,l,:] =  np.fft.fft(xlOFDM[:,l,:])
            
            sigCompUps0[:,l,:] = XRl[:,l,:]#np.concatenate((XX,np.zeros((M,BG),dtype=complex)),1)
            Xdd[:,l,:], Xlp[:,l,:] = papr_reduction(sigCompUps0[:,l,:],V0[:,:,l,:],PNiter,Tau_ss2[l])
            xlOFDMpr0[:,l,:] = np.fft.ifft(Xlp[:,l,:])
            Xldd[:,l,:] =  np.fft.fft(xlOFDMpr0[:,l,:] - xlOFDM[:,l,:])
            xlOFDMpr[:,l,:] = np.fft.ifft(XRl[:,l,:] + Xldd[:,l,:])
            # UEs Received signal

        coeff_IBO_m1dB = np.zeros((M,L),dtype=float)
        xlOFDM_clip = np.zeros((M,L,Mfft),dtype=complex)
        disT = np.zeros((M,L,Mfft),dtype=complex)
        DisFreq = np.zeros((M,L,Mfft),dtype=complex)
        XRamp = np.zeros((M,L,Mfft),dtype=complex)
        for l in range(0,L):    
            for m in range(0,M):
                amax = np.sqrt(np.mean(np.abs(xlOFDMpr[m,l,:])**2)*(10 ** (IBO / 10)))
                xlOFDM_clip[m,l,:] = limiter(xlOFDMpr[m,l,:], 0, amax)
                disT[m,l,:] = xlOFDM_clip[m,l,:] - xlOFDMpr[m,l,:]
            # OFDm demodulation
            XRamp[:,l,:] =  np.fft.fft(xlOFDM_clip[:,l,:])
            DisFreq[:,l,:] = np.fft.fft(disT[:,l,:]) 


        XRl2 = np.zeros((M,L,Mfft),dtype=complex)
        Xldd2 = np.zeros((M,L,Mfft),dtype=complex)
        for l in range(0,L):    
            for m in range(0,M):
                XRl2[m,l,:] = Coef_p[m,l]*XRamp[m,l,:]
                Xldd2[m,l,:] = Coef_p[m,l]*Xldd[m,l,:] 
        
        yk0 = np.zeros((K,Mfft),dtype=complex)
        dr0 = np.zeros((K,Mfft),dtype=complex)
        ds1 = np.zeros((K,Mfft),dtype=complex)
        for mf in range(0,Mfft):
            for l in range(0,L):
                yk0[:,mf] = yk0[:,mf] + np.conjugate(np.transpose(realHFreq3[:,:,l,mf])).dot(XRl2[:,l,mf])
                dr0[:,mf] = dr0[:,mf] + np.conjugate(np.transpose(realHFreq3[:,:,l,mf])).dot(Xldd2[:,l,mf])
                ds1[:,mf] = ds1[:,mf] + np.conjugate(np.transpose(realHFreq3[:,:,l,mf])).dot(DisFreq[:,l,mf])
        for kin in range(0,K):
            for l in range(0,L):
                WHk[l,kin,iNreal] = np.conjugate(np.transpose(realHFreq3[:,kin,l,SCOIindex])).dot(Wlk[:,kin,l,SCOIindex])*np.sqrt(RHOlk[l,kin])
                HDr[l,kin,iNreal] = np.reshape(np.conjugate(np.transpose(realHFreq3[:,kin,l,SCOIindex])),(1,M)).dot(Xldd2[:,l,SCOIindex])
                HDs[l,kin,iNreal] = np.reshape(np.conjugate(np.transpose(realHFreq3[:,kin,l,SCOIindex])),(1,M)).dot(DisFreq[:,l,SCOIindex])
            
            for l in range(0,L):
                for tU in range(0,K):
                    if (tU != kin):
                        WHkt[l,kin,iNreal,tU] =  np.conjugate(np.transpose(realHFreq3[:,kin,l,SCOIindex])).dot(Wlk[:,tU,l,SCOIindex])*np.sqrt(RHOlk[l,tU])
        
        nk = np.sqrt(wp/(2))*(np.random.randn(K,Mfft) + 1j*np.random.randn(K,Mfft))
        yk[:,:,iNreal] = np.reshape(yk0 + nk,(K,Mfft)) + 0*dr0
        dr[:,:,iNreal] = dr0
        ds[:,:,iNreal] = ds1

    
    json_file_path = 'Performance_data.json'

    # Initialiser la liste pour stocker les données
    performance_data = []

    # Ajouter les nouvelles données à la liste
    for kin in range(0, K):
        CPk0 = 0
        PUk0 = 0
        UIkt0 = 0
        HDr0 = 0
        HDs0 = 0
        for l in range(0, L):
            CPk0 = CPk0 + np.mean(G * WHk[l, kin, :])
            PUk0 = PUk0 + G * WHk[l, kin, :] - np.mean(G * WHk[l, kin, :])
            UIkt0 = UIkt0 + G * WHkt[l, kin, :, :]
            HDr0 = HDr0 + (HDr[l, kin, :])
            HDs0 = HDs0 + (HDs[l, kin, :])

        CPksim = (np.abs(CPk0))**2
        PUksim = np.mean((np.abs(PUk0))**2)
        UIktsim = np.sum(np.mean((np.abs(UIkt0))**2, 0))
        HDrsim = np.mean(np.abs(G * HDr0)**2)
        HDssim = np.mean(np.abs(G * HDs0)**2)
        SINRksim0 = CPksim / (PUksim + UIktsim + wp)
        SINRksim = CPksim / (PUksim + UIktsim + HDrsim + HDssim + wp)#avec distorsion

        SEksim0[kin, iNsnap] = xi * (1 - (Tau_p / Tau_c)) * np.log2(1 + SINRksim0)
        SEksim[kin, iNsnap] = xi * (1 - (Tau_p / Tau_c)) * np.log2(1 + SINRksim)

        # Ajouter les nouvelles données au fichier JSON
        performance_data.append({
            "kin": kin,
            "SINRksim": SINRksim0,
            "SEksim": SEksim0[kin, iNsnap]
        })

        # Affichage corrigé des SINR et SE
        print(f"SINR PZF simulé de l'utilisateur (num={kin+1}): {SINRksim0:.4f}")
        print(f"SE PZF simulé de l'utilisateur (kin={kin+1}): {SEksim0[kin, iNsnap]:.4f}")
        print("----------------------")

    # Créer ou réécrire les données dans le fichier JSON (en utilisant 'w' pour écraser le fichier)
    with open(json_file_path, 'w') as json_file:
        json.dump(performance_data, json_file, indent=4)

    
    # # First Part: Plot CDF for SEksim0 and SEksim
    histo, bin_edges_o = np.histogram(np.reshape(SEksim0, (1, K * Nsnap)), 100)
    histo = histo / (K * Nsnap)
    to = bin_edges_o

    histo1, bin_edges_o1 = np.histogram(np.reshape(SEksim, (1, K * Nsnap)), 100)
    histo1 = histo1 / (K * Nsnap)
    to1 = bin_edges_o1

    plt.figure()
    plt.plot(to[:100], np.cumsum(histo), 'r.', label='Original SE')
    plt.plot(to1[:100], np.cumsum(histo1), 'b.', label='Modified SE')
    plt.axis([0, 8, 0, 1])
    plt.grid()
    plt.legend()
    plt.title('CDF of Spectral Efficiency (SE)')
    plt.xlabel('SE')
    plt.ylabel('CDF')
    # plt.show()

    # # -----------------------------------------
    # # Second Part: Compute and plot PAPR CCDF
    # Paramètre de clipping
    np.random.seed(0)
    xlOFDM = (np.random.randn(M, L, N) + 1j * np.random.randn(M, L, N)) / np.sqrt(2)

   
    
    xlOFDMpr = np.copy(xlOFDM)
    nb_clipped_total = 0
    for m in range(M):
        for l in range(L):
            signal = xlOFDM[m, l, :]
            amplitude = np.abs(signal)
            clipped_signal = np.where(amplitude > clipping_threshold,
                                    clipping_threshold * signal / amplitude,
                                    signal)
            nb_clipped_total += np.sum(amplitude > clipping_threshold)
            xlOFDMpr[m, l, :] = clipped_signal

    print("Nombre de valeurs clipées :", nb_clipped_total)

    # Calculer le PAPR pour chaque signal
    PAPRorg = np.zeros((L, M), dtype=float)
    PAPRgg = np.zeros((L, M), dtype=float)

    for l in range(L):
        for m in range(M):
            original_power = np.abs(xlOFDM[m, l, :]) ** 2
            clipped_power = np.abs(xlOFDMpr[m, l, :]) ** 2
            PAPRorg[l, m] = np.max(original_power) / np.mean(original_power)
            PAPRgg[l, m] = np.max(clipped_power) / np.mean(clipped_power)

    # Mise en forme 1D
    PAPRorg_dB = 10 * np.log10(PAPRorg.flatten())
    PAPRgg_dB = 10 * np.log10(PAPRgg.flatten())

    # Tri pour courbe CCDF
    PAPRorg_sorted = np.sort(PAPRorg_dB)
    PAPRgg_sorted = np.sort(PAPRgg_dB)

    ccdf_org = 1.0 - np.arange(1, len(PAPRorg_sorted) + 1) / len(PAPRorg_sorted)
    ccdf_gg = 1.0 - np.arange(1, len(PAPRgg_sorted) + 1) / len(PAPRgg_sorted)

    # Affichage des PAPR moyens
    print("PAPR moyen original (dB):", np.mean(PAPRorg_dB))
    print("PAPR moyen modifié (dB):", np.mean(PAPRgg_dB))

    # Tracer la CCDF
    plt.figure(figsize=(8, 6))
    plt.semilogy(PAPRorg_sorted, ccdf_org, 'r-', label='Original')
    plt.semilogy(PAPRgg_sorted, ccdf_gg, 'b-', label='Avec Clipping')
    plt.grid(True, which='both')
    plt.xlabel('PAPR (dB)')
    plt.ylabel('CCDF (Probabilité)')
    plt.title('CCDF du PAPR avec et sans Clipping')
    plt.legend()
    plt.tight_layout()
     # # -----------------------------------------
    # # Third Part: Compute CDF of SE
    # 1. Aplatir les matrices SEksim0 (sans distorsions) et SEksim (avec distorsions)
    SE_no_distortion = SEksim0.flatten()  # SE sans HDr et HDs
    SE_with_distortion = SEksim.flatten()  # SE avec HDr et HDs

    # 2. Nettoyer les valeurs (enlever zéros et NaNs)
    SE_no_distortion = SE_no_distortion[SE_no_distortion > 0]
    SE_no_distortion = SE_no_distortion[~np.isnan(SE_no_distortion)]

    SE_with_distortion = SE_with_distortion[SE_with_distortion > 0]
    SE_with_distortion = SE_with_distortion[~np.isnan(SE_with_distortion)]

    # 3. Trier les valeurs
    SE_no_sorted = np.sort(SE_no_distortion)
    SE_with_sorted = np.sort(SE_with_distortion)

    # 4. Calculer les CDFs
    cdf_no = np.arange(1, len(SE_no_sorted) + 1) / len(SE_no_sorted)
    cdf_with = np.arange(1, len(SE_with_sorted) + 1) / len(SE_with_sorted)
 
    # 5. Tracer la courbe
    plt.figure(2)
    # plt.figure(figsize=(8, 5))
    plt.plot(SE_no_sorted, cdf_no, label="SE sans distorsions (HDr/HDs)", color='blue', linewidth=2)
    plt.plot(SE_with_sorted, cdf_with, label="SE avec distorsions (HDr/HDs)", color='orange', linestyle='--', linewidth=2)

    plt.xlabel("Spectral Efficiency (bit/s/Hz)")
    plt.ylabel("CDF")
    plt.title("CDF du Spectral Efficiency - PZF (simulé)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()




    

    
    # def empirical_cdf(data):
    #     """
    #     Calcule la CDF empirique d'un vecteur de données.
    #     Retourne les valeurs triées et leur CDF.
    #     """
    #     data = np.asarray(data).flatten()  # s'assurer que c'est un vecteur 1D
    #     if data.size == 0:
    #         raise ValueError("La donnée est vide. Impossible de calculer la CDF.")
    #     sorted_data = np.sort(data)
    #     cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    #     return sorted_data, cdf

    # # Vérifie que tu as bien plusieurs données (à adapter avec tes vraies données)
    # print("SEkTheo shape:", np.shape(SE_individuel_vecteur))
    # print("SEksim0 shape:", np.shape(SEksim0))

    # # Aplatir s'il le faut
    # SE_individuel_vecteuro = np.asarray(SE_individuel_vecteur).flatten()
    # SEksim0 = np.asarray(SEksim0).flatten()

    # # Calcul CDF
    # SEth_sorted, cdf_theo = empirical_cdf(SE_individuel_vecteur)
    # SEksim_sorted, cdf_sim = empirical_cdf(SEksim0)

    # # Tracer les courbes CDF
    # plt.figure(figsize=(9, 6))
    # plt.plot(SEksim_sorted, cdf_sim, label='SE simulé (SEksim0)', color='blue', linewidth=2)
    # plt.plot(SEth_sorted, cdf_theo, label='SE théorique (SE_total)', color='red', linestyle='--', linewidth=2)

    # plt.xlabel("Spectral Efficiency (bits/s/Hz)")
    # plt.ylabel("CDF")
    # plt.title("Comparaison entre SE théorique et simulé")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    plt.show()

# %%
