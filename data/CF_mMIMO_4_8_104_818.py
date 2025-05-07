import os

NUM_CORES="2"
os.environ["OMP_NUM_THREADS"] = NUM_CORES
os.environ["OPENBLAS_NUM_THREADS"] = NUM_CORES
os.environ["MKL_NUM_THREADS"] = NUM_CORES
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_CORES
os.environ["NUMEXPR_NUM_THREADS"] = NUM_CORES

from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

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


def design_precoder_local_PFZF(H3barl,Thetalk):
    W_PFZFl = np.zeros((M,M-1,L,Mfft),dtype=complex)
    W_MRTl = np.zeros((M,Tau_p,L,Mfft),dtype=complex)
    Wlk = np.zeros((M,Tau_p,L,Mfft),dtype=complex)
    
    for l in range(0,L):
        for mf in range(0,Mfft):
            W_PFZFl[:,:Tau_ss1[l],l,mf] =  H3barl[:,:,l,mf].dot(ESl[l,:,:Tau_ss1[l]]).dot(np.linalg.inv(np.transpose(np.conjugate(ESl[l,:,:Tau_ss1[l]])).dot(np.transpose(np.conjugate(H3barl[:,:,l,mf]))).dot(H3barl[:,:,l,mf]).dot(ESl[l,:,:Tau_ss1[l]]))) #W_FZFl[:,:,l,mf] = (np.linalg.inv((H3barl[:,:,l,mf]).dot(np.transpose(np.conjugate(H3barl[:,:,l,mf]))) + 0*SIGl + Pl)).dot(H3barl[:,:,l,mf]) # 
            W_MRTl[:,:,l,mf] =  H3barl[:,:,l,mf]
        for k in range(0,Tau_ss1[l]):
            ii = int(Ilk[l,k])
            Wlk[:,ii,l,:] =  np.sqrt((M-Tau_ss1[l])*Thetalk[l,ii])*W_PFZFl[:,k,l,:] #W_FZFl[:,k,l,mf]/np.real(K*np.transpose(np.conjugate(H3barl[:,k,l,mf])).dot(W_FZFl[:,k,l,mf])) #
        for k in range(Tau_ss1[l],Tau_p):
            ii = int(Ilk[l,k])
            Wlk[:,ii,l,:] =  W_MRTl[:,ii,l,:]/np.sqrt(M*Thetalk[l,ii])           

    WW = np.zeros((M,(int(K/Tau_p)+1)*Tau_p,L,Mfft),dtype=complex)
    for tp in range(0,int(K/Tau_p)+1):
        WW[:,tp*Tau_p:(tp+1)*Tau_p,:,:] = Wlk

    W_PZFlk = WW[:,:K,:]                

    return W_PZFlk

def design_precoder_local_FZF(H3barl,Thetalk):
    W_FZFl = np.zeros((M,Tau_p,L,Mfft),dtype=complex)
    
    normalization = np.sqrt((M-Tau_p)*Thetalk)
    
    for l in range(0,L):
        for mf in range(0,Mfft):
            W_FZFl[:,:,l,mf] =  H3barl[:,:,l,mf] @ (np.linalg.inv(np.transpose(np.conjugate(H3barl[:,:,l,mf])) @ H3barl[:,:,l,mf])) 
            W_FZFl[:,:,l,mf] = W_FZFl[:,:,l,mf] * normalization[l,:Tau_p]

    WW = np.zeros((M,(int(K/Tau_p)+1)*Tau_p,L,Mfft),dtype=complex)
    for tp in range(0,int(K/Tau_p)+1):
        WW[:,tp*Tau_p:(tp+1)*Tau_p,:,:] = W_FZFl

    W_FZFlk = WW[:,:K,:]                


    return W_FZFlk

def design_precoder_local_MRT(H3barl,Thetalk):
    W_MRTl = np.zeros((M,Tau_p,L,Mfft),dtype=complex)
    
    normalization = 1 / np.sqrt(M*Thetalk)
    for l in range(0,L):
        for mf in range(0,Mfft):
            W_MRTl[:,:,l,mf] =  H3barl[:,:,l,mf]
            W_MRTl[:,:,l,mf] =  W_MRTl[:,:,l,mf]  * normalization[l,:Tau_p]

    WW = np.zeros((M,(int(K/Tau_p)+1)*Tau_p,L,Mfft),dtype=complex)
    for tp in range(0,int(K/Tau_p)+1):
        WW[:,tp*Tau_p:(tp+1)*Tau_p,:,:] = W_MRTl

    W_MRTlk = WW[:,:K,:]                

    return W_MRTlk

def find_indices_UEs_with_same_pilots(userPilotIndex): 
    """
    
    Solution trouvée en ligne : https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
    returne les indices de tout les pilots répétées dans UserPilot_index,
    example : [0,1,2,0,1] retourne trois arrays avec pour indice le numéro du pilot et contenant le set d'utilisateur Pk l'utilisant :
    [0]=0,3    [1] = 1,4    [2] = 2 
    
    """
def find_same_pilot_in_PZF_groups(Ilk,Tau_ss, L):
    
    group_strong_UE = np.zeros((L), dtype=object)
    
    # creates an array of indices, sorted by unique element
    idx_sort = np.argsort(userPilotIndex)

    # sorts records array so all unique elements are together 
    sorted_records_array = userPilotIndex[idx_sort]
    
    # returns the unique values, the index of the first occurrence of a value, and the count for each element
    vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)
    
    # splits the indices into separate arrays
    samePilot_idx_list = np.split(idx_sort, idx_start[1:])

    for l in range(L):
        group_strong_pilots = Ilk[l,:Tau_ss[l]]
        # --- Find and add the UEs with same pilots in both group --- #
        temp_group_strong_UE= np.zeros(np.size(group_strong_pilots), dtype=object) # list formating inside an array of dimension nb_AP (each entry may have different size)

        for index, value in enumerate(group_strong_pilots):
            temp_group_strong_UE[index]= np.where(len(samePilot_idx_list[value]) > 1, samePilot_idx_list[value], value)
            # Replace the current value (pilot index) by a tuple containing the index of All UE sharing this pilot
        if len(temp_group_strong_UE)>0:
            temp_group_strong_UE = np.hstack(temp_group_strong_UE)  # flatten the list where multiple UE index are saved in a tuple as a linear sequence
        else :
            temp_group_strong_UE=[]
        group_strong_UE[l] = temp_group_strong_UE
        
    return group_strong_UE 
        
# ----------------------------------------- #
# ------------ PARAMETERS SYSTEM ---------- # !!!
# ----------------------------------------- #

# ------  OFDM & Sous porteuse
BG = 2
Mfft = 818
Mqam = 4
modulation_type = "qam"

# User grouping related Precoding
# nuTh1 = 0.999999999999999
nuTh1 = 1
#  ------  Topologie & DL/UL ressouces
L = 100
M = 4
K = 8
Tau_p = 8
Tau_c = 168
xi = 0.5

D = 60 # Surface en m² de l'environnement
APh = 10
UEh = 1.5


#  -----  Modèle canal (fading)
sigma_sh_dB = 4
sigma_sh = 10**(sigma_sh_dB/10)
Kau = 0.5

#  -----  Puissances émission DL/UL + noise
Pin = 1 #pin*M
Pinmw = 1000*Pin

wp_dBm = -93
wp = 0.001*(10**(wp_dBm/10))

Rho_l_max_dBm = 10*np.log10(Pinmw) - 0*wp_dBm
Rho_l_max = Pin # 0.001*(10**(Rho_l_max_dBm/10))

pk_dBm = 10*np.log10(100) - 0*wp_dBm
pk = 0.1 #0.001*(10**(pk_dBm/10))

# ------ Paramètres de simulation
Nsnap = 1
Nreal = 1

# ----------------------------------------- #
# ------------ Début Simulation ----------- # !!!
# ----------------------------------------- #
SEkTheo_PZF = np.zeros(Nsnap,dtype=float)
SEksim_PZF = np.zeros(Nsnap,dtype=float)

# ------------- Generating Topology ----------- #
iNsnap=1
for UOIindex in range(0,8):
    for SCOIindex in range(0,818):
    
    # SCOIindex = np.random.randint(BG,Mfft-BG,1)[0]#Selects a subcarrier index within a range, ensuring a margin BG from the start and end
    
        with open("distance_matrices_all_Dlk.json", "r", encoding="utf-8") as file:
            all_distance_matrices = json.load(file)

        # Accéder à Dlk1 et convertir en tableau Nu
        # mPy
        Dlk1 = np.array(all_distance_matrices["Dlk10"])

        # Transposer si nécessaire (100 points x 8 APs au lieu de 8 x 100)
        
        Dlk = Dlk1.T

    # Pilots
        Pilots = np.sqrt(Tau_p)*randU(Tau_p) #Random signals
        # Uniform Pilot assignement
        userPilotIndex0 = list(frange(0,Tau_p,1))
        userPilotIndex00 = [] #10*np.ones((Tau_p*(int(K/Tau_p)+1)),dtype=int)
        for tp in range(0,int(K/Tau_p)+1):
            userPilotIndex00 = np.concatenate((userPilotIndex00,userPilotIndex0))#Assigns pilots to users using a uniform assignment strategy
        userPilotIndex = userPilotIndex00[:K]
        
        PhiP = np.zeros((Tau_p,K),dtype=complex)
        for up in range(0,K):
            PhiP[:,up] = Pilots[:,int(userPilotIndex[up])]#Constructs a pilot matrix PhiP where each column corresponds to the pilot assigned to a user
        
        PL_lk_dB = -30.5 - 36.7*np.log10(Dlk) #Computes path loss (PL) in dB based on distance Dlk
        PL_lk = 10**(PL_lk_dB/10)
           
        #Shadow Fading Model
        al0 = np.random.randn(L,1)
        al = np.repeat(al0,K,-1)
        bk0 = np.random.randn(K,1)
        bk00 = np.repeat(bk0,L,-1)
        bk = np.transpose(bk00)
        zlk = np.sqrt(Kau)*al + np.sqrt(1-Kau)*bk
        sigma_sh_Zlk = sigma_sh*zlk
        BETAlk = PL_lk*(10**(sigma_sh_Zlk/10)) #the large-scale fading matrix
        
        # SORT BETAl
        # Sorting and Pilot Contamination Reduction
        Itp = np.eye(Tau_p)
        Ilk = np.zeros((L,Tau_p),dtype=int)
        
        BETAilk = np.zeros((L,Tau_p),dtype=float) 
        for l in range(0,L):
            for tp in range(0,Tau_p):
                BETAilk[l,tp] = max(BETAlk[l,tp:K:Tau_p])#Finds the strongest beta values for each tp                
            tmp = np.argsort(BETAilk[l,:])#Sorts them in descending order
            Ilk[l,:] = tmp[::-1]

        #Initializes variables for pilot contamination mitigation
        BETAblk = np.zeros((L,Tau_p), dtype=float)
        Tau_ss1 = np.zeros((L), dtype=int)
        TausMax = min(M-1,Tau_p)
        
        #Selects Tau_ss1 as the number of significant pilot symbols
        for l in range(0,L):
            BETAblk[l,:] = BETAilk[l,Ilk[l,:]]
            Tau_ss1[l] = TausMax
            if nuTh1 == 0 :
                Tau_ss1[l] = 0
            else :
                for ts in range(1,TausMax):
                    if ((np.sum(BETAblk[l,:ts])/np.sum(BETAblk[l,:]))>nuTh1):
                        Tau_ss1[l] = ts
                        break

        #ESl, a matrix storing pilot sequences for different users.
        ESl = np.zeros((L,Tau_p,TausMax),dtype=int)        
        for l in range(0,L):
            for ts in range(0,Tau_ss1[l]):
                ii = int(Ilk[l,ts])
                ESl[l,:,ts] = Itp[:,ii]
        group_strong_UE = find_same_pilot_in_PZF_groups(Ilk,Tau_ss1, L)#Determines strong user groups with the same pilot sequence
        
        Pk = pk*np.diag(np.ones(K))
        
        # Channel estimation
        Clk = np.zeros((L,K),dtype=float)#Clk estimates the channel coefficients for each user at different BSs.
        for k in range(0,K):
            UONIindex = [] # np.zeros((1,1),dtype=int)
            for kk in range(0,K):
                if (userPilotIndex[kk]==userPilotIndex[k]) & (kk!=k):
                    UONIindex = np.concatenate((UONIindex,np.reshape(kk,(1))),0)#identifies interfering users (UONIindex) that have the same pilot.
            for l in range(0,L):
                betalt = 0
                for ind in range(0,len(UONIindex)):
                    betalt = betalt + BETAlk[l,int(UONIindex[ind])]#channel gain correction using large-scale fading coefficients BETAlk.
                Clk[l,k] = (np.sqrt(pk)*BETAlk[l,k])/((Tau_p*pk*(betalt+BETAlk[l,k])) + wp)
        #Power allocation
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
        # Power Control Optimization
        RHOlk = np.zeros((L,K),dtype=float)
        for l in range(0,L):
            for k in range(0,K):
                RHOlk[l,k] = ((NUlk[l,k]/np.sum(NUlk[l,:]))*Rho_l_max)    # # Power control
        

        # Simulation
        yk = np.zeros((K,Mfft,Nreal),dtype=complex)
        WHk_PZF = np.zeros((L,Nreal),dtype=complex)
        WHkt_PZF = np.zeros((L,Nreal,K),dtype=complex)
        WHk_DAC_PZF = np.zeros((L,Nreal),dtype=complex)
        WHkt_DAC_PZF = np.zeros((L,Nreal,K),dtype=complex)


        d_sample_sim = np.zeros((M,L,Nreal),dtype=complex)
        Distortion_DAC_PZF = np.zeros((Nreal, L),dtype=complex) 
        Distortion_DAC_FZF = np.zeros((Nreal, L),dtype=complex) 
        Distortion_DAC_MRT = np.zeros((Nreal, L),dtype=complex) 
        # BITk = []
        Qk = np.zeros((Nreal),dtype=complex)
        print("Now In 'Simulation'... --" )
        svg_realHFreq3 = np.zeros((Nreal,M,K,L,Mfft),dtype=complex)
        for iNreal in range(0,Nreal):  
        
            matrix_path = os.path.join(os.path.dirname(__file__), 'matrix_4_8_104_818.npy')


            H3 = np.load(matrix_path)  # Shape (4, 8, 52, 818)
            H3= H3[:,:,:100,:] 
            # print("Shape de H3 :", H3.shape)

            # channel in frequency domain
            realHFreq3 = H3
            
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
            # ------ PRECODER DESIGNS ------ #
            Wlk_PZF = design_precoder_local_PFZF(H3barl,Thetalk) # Wlk = Précodeur PZF

            
            # ------ BIT GENERATION ------ #
            bitk = np.random.randint(2,size=int(np.log2(Mqam)*K*Mfft))
            qk = SignalTools.mapping(bitk, Mqam, modulation_type)
            qk = qk/np.sqrt(np.var(qk))
            qkP = np.reshape(qk,(K,Mfft)) 
            Qk[iNreal] = qkP[UOIindex,SCOIindex] #np.concatenate((Qk,qk[UOIindex]))

            Xl_PZF = np.zeros((M,L,Mfft),dtype=complex)
        
            XlOFDM_PZF = np.zeros((M,L,Mfft),dtype=complex)

            # --  Application precoder sur les signaux  -- #
            for l in range(0,L):
                for mf in range(0,Mfft):              
                    Xl_PZF[:,l,mf] = Wlk_PZF[:,:,l,mf].dot(np.diag(np.sqrt(RHOlk[l,:]))).dot(qkP[:,mf])
                Xl_PZF[:,l,:BG] = np.zeros((M,BG),dtype=float)
                Xl_PZF[:,l,Mfft-BG:] = np.zeros((M,BG),dtype=float)
                
                # -- OFDM modulation -- #
                XlOFDM_PZF[:,l,:] = np.fft.ifft(Xl_PZF[:,l,:])

            #-------- SANS DAC  ---------#
            # PZF ---> 
            for l in range(0,L):
                WHk_PZF[l,iNreal] = np.conjugate(np.transpose(realHFreq3[:,UOIindex,l,SCOIindex])).dot(Wlk_PZF[:,UOIindex,l,SCOIindex])*np.sqrt(RHOlk[l,UOIindex])
                for tU in range(0,K):
                    if (tU != UOIindex):
                        WHkt_PZF[l,iNreal,tU] =  np.conjugate(np.transpose(realHFreq3[:,UOIindex,l,SCOIindex])).dot(Wlk_PZF[:,tU,l,SCOIindex])*np.sqrt(RHOlk[l,tU])
    
        
        Rnd_Ant = int(np.random.randint(M))
        Rnd_AP = int(np.random.randint(L))
        # ----- Calcul performance SIMULEES ----- #   
        # PZF --->    
        CPk0_DAC = 0
        PUk0_DAC = 0
        UIkt0_DAC = 0
        CPk0 = 0
        PUk0 = 0
        UIkt0 = 0
        
        total_distortion = np.sum(Distortion_DAC_PZF, axis =1) # Axis 1 = APs
        HWI_sim = np.mean(np.abs(total_distortion)**2, axis =0)  


        for l in range(0,L):
            # AVEC DAC
            CPk0_DAC = CPk0_DAC + np.mean(WHk_DAC_PZF[l,:])
            PUk0_DAC = PUk0_DAC + WHk_DAC_PZF[l,:] - np.mean(WHk_DAC_PZF[l,:])
            UIkt0_DAC = UIkt0_DAC + WHkt_DAC_PZF[l,:,:]
            # SANS DAC
            CPk0 = CPk0 + np.mean(WHk_PZF[l,:])
            PUk0 = PUk0 + WHk_PZF[l,:] - np.mean(WHk_PZF[l,:])
            UIkt0 = UIkt0 + WHkt_PZF[l,:,:]
        CPksim_DAC = (np.abs(CPk0_DAC))**2
        PUksim_DAC = np.mean(np.abs(PUk0_DAC)**2)
        UIktsim_DAC = np.sum(np.mean((np.abs(UIkt0_DAC))**2,0))
        
        CPksim = (np.abs(CPk0))**2
        PUksim = np.mean((np.abs(PUk0))**2)
        UIktsim = np.sum(np.mean((np.abs(UIkt0))**2,0))
        print("\r-------------")
        # Sim SANS DAC
        # print("UOIindex",UOIindex)# %%
        SINRksim_PZF = CPksim / (PUksim + UIktsim + wp)
        # print(" SINR PZF simulé :",SINRksim_PZF)
        SEksim_PZF = xi*(1-(Tau_p/Tau_c))*np.log2(1+SINRksim_PZF)
        # print("SE PZF simulé : " +str(SEksim_PZF))
        
        
        
        # print("SE PZF théorique : " +str(SEkTheo_PZF))
        data_to_save = {
            "UOIindex": UOIindex,
            "SINR_simule_PZF": SINRksim_PZF,
            "SE_simule_PZF": SEksim_PZF,
            
        }

        # Nom du fichier JSON
        file_name = "matrix2_4_8_104_818/performance_4_8_104_818.json"

        # Lire les anciennes données du fichier, si elles existent
        try:
            with open(file_name, 'r') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            existing_data = []

        # Ajouter les nouvelles données
        existing_data.append(data_to_save)

        # Enregistrer les nouvelles données dans le fichier JSON
        with open(file_name, 'w') as file:
            json.dump(existing_data, file, indent=4)

        # Afficher les résultats
        print("UOIindex", UOIindex)
        print("SCOIindex", SCOIindex)
        print("SINR PZF simulé :", SINRksim_PZF)
        print("SE PZF simulé : " + str(SEksim_PZF))
    

# Génère un faux vecteur de SE pour tester

SE_sim_list = [entry["SE_simule_PZF"] for entry in existing_data]

# Tri
SE_sim_sorted = np.sort(SE_sim_list)

# CDF
cdf_sim = np.arange(1, len(SE_sim_sorted)+1) / len(SE_sim_sorted)

# Tracé
plt.figure()
plt.plot(SE_sim_sorted, cdf_sim, label='SE simulé')

plt.xlabel("Spectral Efficiency (bit/s/Hz)")
plt.ylabel("CDF")
plt.title("CDF de l'efficacité spectrale")
plt.legend()
plt.grid(True)
plt.show()