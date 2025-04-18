import json
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import folium
import scipy
from scipy.io import savemat
import matplotlib.pyplot as plt

# Charger les points d'accès (AP) et les utilisateurs (UE) à partir des fichiers JSON
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def gram_schmidt(M):
    """Perform Gram-Schmidt orthogonalization process on a matrix."""
    M = M.copy()
    n = M.shape[1]
    for j in range(n):
        for k in range(j):
            M[:, j] -= np.dot(M[:, k], M[:, j]) * M[:, k]
        norm = np.linalg.norm(M[:, j])
        if norm > 1e-10:
            M[:, j] /= norm
    return M

def randU(n):
    """Generate a random unitary matrix of size n×n."""
    X = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, R = np.linalg.qr(X)
    D = np.diag(np.sign(np.diag(R)))
    return Q @ D
        
 
def generate_vector(L, L0):
    """Generate a binary vector with L0 zeros and (L-L0) ones"""
    if L0 > L:
        raise ValueError("L0 ne peut pas être supérieur à L.")
    
    vector = np.zeros(L)
    vector[:L-L0] = 1  
    np.random.shuffle(vector)
    
    return vector 

# Charger les données
ap_file_path = "coordinates_of_100_AP.json"
ue_file_path = "AP_coordinates.json"

access_points = load_json(ap_file_path)
users = load_json(ue_file_path)

# Fonction pour calculer la distance Haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Rayon de la Terre en km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c * 1000
# Cell-free System model parameters
Mqam = 16  # Modulation order (16-QAM)
modulation_type = "qam"  # Modulation type
L = 100  # Number of access points (APs)
K =8   # Number of user equipment (UEs)
M = 16  # Number of antennas per AP
Tau_p = 8  # Number of pilot symbols
Tau_c = 168  # Coherence block length in symbols
xi = 0.5  # System inefficiency factor
sigma_sh_dB = 4  # Shadow fading standard deviation in dB
sigma_sh = 10**(sigma_sh_dB/10)  # Convert shadow fading from dB to linear scale
Kau = 0.5  # Parameter controlling spatial correlation in shadow fading
APh = 10  # AP height in meters
UEh = 1.5  # UE height in meters

# Power consumption parameters
mul = 0.5  # Power amplifier efficiency
Pcl = 0.1  # Circuit power consumption per antenna [W]
P0l = 0.1  # Fixed power consumption per AP [W]
Pbtl = 0.25*1e-9  # Backhaul power consumption coefficient [J/bit]

# Channel parameters
FcGHz = 2.6  # Carrier frequency in GHz
Bw = 20e6  # Bandwidth in Hz
noiseFigure = 7  # Noise figure in dB

# Calculate noise power
wp_dBm = -174 + 10*np.log10(Bw) + noiseFigure  # Thermal noise power in dBm
wp = 0.001*(10**(wp_dBm/10))  # Convert to linear scale [W]
 
# Maximum transmit power per AP
Rho_l_max_dBm = 10*np.log10(200) - wp_dBm  # Maximum power in dBm relative to noise
Rho_l_max = (10**(Rho_l_max_dBm/10))  # Convert to linear scale
 
# Pilot transmit power
pk_dBm = 10*np.log10(100) - wp_dBm  # Pilot power in dBm relative to noise
pk = (10**(pk_dBm/10))  # Convert to linear scale
 
# Simulation parameters
Nsnap = 1  # Number of snapshots (network realizations)
Nreal = 1000  # Number of channel realizations per snapshot



# Initialize result arrays
SEkTheo = np.zeros((K,Nsnap,Nreal,L),dtype=float)  # Spectral efficiency per user
sumSEkTheo = np.zeros((Nsnap,Nreal,L),dtype=float)  # Sum spectral efficiency
EE = np.zeros((Nsnap,Nreal,L),dtype=float)  # Energy efficiency
Ptot = np.zeros((Nsnap,Nreal,L),dtype=float)  # Total power consumption
# Vérifier la cohérence des données
if len(access_points) != 100 or len(users) != 8:
    raise ValueError("Erreur: Le fichier JSON ne contient pas 100 AP ou 8 utilisateurs.")

# Extraire les coordonnées
APcoords = np.array([[ap["latitude"], ap["longitude"]] for ap in access_points])
UEcoords = np.array([[ue["latitude"], ue["longitude"]] for ue in users])

APcoords_rad = np.radians(APcoords)
UEcoords_rad = np.radians(UEcoords)

# Initialiser la matrice de distances Dlk (8 utilisateurs × 100 AP)
Dlk = np.zeros((len(users), len(access_points)), dtype=float)

# Calculer la matrice de distances
for k, ap in enumerate(access_points):
    ap_lat, ap_lon = ap["latitude"], ap["longitude"]
    for l, ue in enumerate(users):
        ue_lat, ue_lon = ue["latitude"], ue["longitude"]
        Dlk[l, k] = haversine(ap_lat, ap_lon, ue_lat, ue_lon)

# Afficher la matrice des distances
print(Dlk.shape)

# Main simulation loop
for iNsnap in range(0,Nsnap):
    # Calculate distance matrix between all APs and UEs

    Pilots = np.sqrt(Tau_p)*randU(Tau_p)    
    
    # Uniform pilot assignment to users (pilot reuse)
    userPilotIndex0 = np.arange(Tau_p)  # Safer way to generate indices
    userPilotIndex00 = np.tile(userPilotIndex0, (K // Tau_p) + 1)[:K]  # Avoid explicit concatenation

    for tp in range(0,int(K/Tau_p)+1):
        # Repeat the pilot indices enough times to cover all users
        userPilotIndex00 = np.concatenate((userPilotIndex00,userPilotIndex0))
    userPilotIndex = userPilotIndex00[:K]  # Keep only as many indices as there are users
    
    # Assign pilots to users
    PhiP = np.zeros((Tau_p,K),dtype=complex)
    for up in range(0,K):
        PhiP[:,up] = Pilots[:,int(userPilotIndex[up])]
    
    # Calculate path loss between each AP and UE (in dB)
    PL_lk_dB = -36.7*np.log10(Dlk) - 22.7 - 26*np.log10(FcGHz)  # Path loss model
    PL_lk = 10**(PL_lk_dB/10)  # Convert from dB to linear scale
    
    # Generate spatially correlated shadow fading
    al0 = np.random.randn(L,1)  # AP-specific component
    al = np.repeat(al0,K,-1)  # Repeat for each UE
    bk0 = np.random.randn(K,1)  # UE-specific component
    bk00 = np.repeat(bk0,L,-1)  # Repeat for each AP
    bk = np.transpose(bk00)
    # Combine AP and UE components with correlation parameter Kau
    zlk = np.sqrt(Kau)*al + np.sqrt(1-Kau)*bk
    sigma_sh_Zlk = sigma_sh*zlk  # Scale by shadow fading standard deviation
    
    # Calculate channel gain including path loss and shadow fading
    BETAlk = PL_lk*(10**(sigma_sh_Zlk/10))
    
    # Create diagonal matrix of pilot transmit powers
    Pk = pk*np.diag(np.ones(K))
    
    # Channel estimation - calculate channel estimation coefficients
    Clk = np.zeros((L,K),dtype=float)
    for k in range(0,K):
        # Find users with same pilot as user k (pilot contamination)
        UONIindex = []  # Users of nOn-Interest with same pilot
        for kk in range(0,K):
            if (userPilotIndex[kk]==userPilotIndex[k]) & (kk!=k):
                UONIindex = np.concatenate((UONIindex,np.reshape(kk,(1))),0)
        
        # Calculate channel estimation coefficient for each AP-UE pair
        for l in range(0,L):
            betalt = 0  # Sum of channel gains from interfering users
            for ind in range(0,len(UONIindex)):
                betalt = betalt + BETAlk[l,int(UONIindex[ind])]
            # MMSE estimation coefficient
            Clk[l,k] = (np.sqrt(pk)*BETAlk[l,k])/((Tau_p*pk*(betalt+BETAlk[l,k])) + 1)
    
    # Calculate normalized channel gain for each AP-UE pair
    NUlk = np.zeros((L,K),dtype=float)
    for k in range(0,K):
        # Find users with same pilot as user k (pilot contamination)
        UONIindex = []
        for kk in range(0,K):
            if (userPilotIndex[kk]==userPilotIndex[k]) & (kk!=k):
                UONIindex = np.concatenate((UONIindex,np.reshape(kk,(1))),0)
        
        # Calculate normalized channel gain
        for l in range(0,L):
            betalt = 0
            for ind in range(0,len(UONIindex)):
                betalt = betalt + BETAlk[l,int(UONIindex[ind])]
            # Normalized channel gain considering pilot contamination
            NUlk[l,k] = ((pk*Tau_p)*(BETAlk[l,k]**2))/((pk*Tau_p*(betalt+BETAlk[l,k])) + 1)
            
    # Calculate Theta parameter (ratio of normalized gain to square of estimation coefficient)
    Thetalk = NUlk/(Clk**2)
    
    # Initialize performance metric arrays
    LB_sumSE = np.zeros(L,dtype=float)  # Lower bound on sum spectral efficiency
    UB_sumSE = np.zeros(L,dtype=float)  # Upper bound on sum spectral efficiency
    Avg_sumSE = np.zeros(L,dtype=float)  # Average sum spectral efficiency
    LB_EE = np.zeros(L,dtype=float)  # Lower bound on energy efficiency
    UB_EE = np.zeros(L,dtype=float)  # Upper bound on energy efficiency
    Avg_EE = np.zeros(L,dtype=float)  # Average energy efficiency
    
    # Simulate different numbers of active APs
    for L0 in range(0,L):
        print(L0)  # Print progress
        # Simulate multiple channel realizations
        for iNreal in range(0,Nreal):
            # Generate binary vector indicating which APs are active (1) or inactive (0)
            # L0 is the number of inactive APs, so L-L0 APs are active
            vi = generate_vector(L, L0)
            
            # Power allocation for each AP-UE pair
            RHOlk = np.zeros((L,K),dtype=float)
            for l in range(0,L):
                for k in range(0,K):
                    # Power control: allocate power based on channel quality
                    # vi[l] is binary (0/1) indicating if AP l is active
                    # Allocate power proportional to normalized channel gain
                    RHOlk[l,k] = vi[l]*((NUlk[l,k]/np.sum(NUlk[l,:]))*Rho_l_max)
            
            # Calculate SINR and spectral efficiency for each user
            for k in range(0,K):
                UOIindex = k  # User Of Interest
                UONIindex = []  # Users with same pilot as user k
                for kk in range(0,K):
                    if (userPilotIndex[kk]==userPilotIndex[UOIindex]) & (kk!=UOIindex):
                        UONIindex = np.concatenate((UONIindex,np.reshape(kk,(1))),0)
                
                # Sum of channel gains from interfering users (not used in this loop)
                betalt = 0
                for ind in range(0,len(UONIindex)):
                    betalt = betalt + BETAlk[l,int(UONIindex[ind])]
                
                # Calculate coherent processing gain (desired signal power)
                CPktheo0 = np.sqrt(RHOlk[:,UOIindex]*NUlk[:,UOIindex])
                CPktheo = (M-Tau_p)*(np.sum(CPktheo0))**2
                
                # Calculate pilot contamination interference
                PUktheo0 = 0
                for ind in range(0,len(UONIindex)):
                    # Sum interference from users with same pilot
                    PUktheo0 = PUktheo0 + (np.sum(np.sqrt(RHOlk[:,int(UONIindex[ind])]*NUlk[:,UOIindex])))**2
                PUktheo = (M-Tau_p)*PUktheo0
                
                # Calculate non-coherent interference from all users
                UIkttheo0 = 0
                for t in range(0,K):
                    UIkttheo0 = UIkttheo0 + RHOlk[:,t]*(BETAlk[:,UOIindex]-NUlk[:,UOIindex])   
                UIkttheo = np.sum(UIkttheo0)
                
                # Calculate SINR (Signal-to-Interference-plus-Noise Ratio)
                SINRktheo = CPktheo/(PUktheo + UIkttheo + 1)
                
                # Calculate spectral efficiency in Mbps
                # xi: system inefficiency factor
                # (1-(Tau_p/Tau_c)): overhead due to pilots
                # (Bw/1e6): convert from bps/Hz to Mbps
                SEkTheo[k,iNsnap,iNreal,L0] = xi*(1-(Tau_p/Tau_c))*(Bw/1e6)*np.log2(1+SINRktheo)
                
            # Calculate sum spectral efficiency (total network throughput)
            sumSEkTheo[iNsnap,iNreal,L0] = np.sum(SEkTheo[:,iNsnap,iNreal,L0])
            
            # Compute total power consumption
            Pl = np.zeros((L),dtype=float)  # Power consumption of each AP
            Pfl = np.zeros((L),dtype=float)  # Fixed power consumption
            for l in range(0,L):
                # AP power: transmit power + circuit power for active APs
                Pl[l] = (1/mul)*wp*M*np.sum(RHOlk[l,:]*NUlk[l,:]) + vi[l]*M*Pcl
                # Fixed power + backhaul power proportional to data rate
                Pfl[l] = P0l + Pbtl*sumSEkTheo[iNsnap,iNreal,L0]
            
            # Total power consumption across all APs
            Ptot[iNsnap,iNreal,L0] = np.sum(Pl) + np.sum(Pfl)
            
            # Calculate energy efficiency (bits/Joule)
            EE[iNsnap,iNreal,L0] = sumSEkTheo[iNsnap,iNreal,L0]/Ptot[iNsnap,iNreal,L0]

        # Calculate statistics across all channel realizations
        LB_sumSE[L0] = np.min(sumSEkTheo[iNsnap,:,L0])  # Minimum (lower bound) of sum SE
        UB_sumSE[L0] = np.max(sumSEkTheo[iNsnap,:,L0])  # Maximum (upper bound) of sum SE
        Avg_sumSE[L0] = np.median(sumSEkTheo[iNsnap,:,L0])  # Median (average) of sum SE
        
        LB_EE[L0] = np.min(EE[iNsnap,:,L0])  # Minimum (lower bound) of EE
        UB_EE[L0] = np.max(EE[iNsnap,:,L0])  # Maximum (upper bound) of EE
        Avg_EE[L0] = np.median(EE[iNsnap,:,L0])  # Median (average) of EE
        
# Plot performance results

# Plot Sum Spectral Efficiency vs number of active APs (L1)
plt.figure()
# Plot lower bound, median, and upper bound
plt.plot(L - np.arange(0,L), LB_sumSE.transpose(), 'b.', label='Lower Bound')
plt.plot(L - np.arange(0,L), Avg_sumSE.transpose(), 'g.', label='Median (Avg)')
plt.plot(L - np.arange(0,L), UB_sumSE.transpose(), 'r.', label='Upper Bound')
plt.axis([1, 87, 20, 2000])  # Set axis limits
plt.ylabel("SE [Mbit/s]")  # Y-axis label
plt.xlabel("L1")  # X-axis label (number of active APs)
plt.legend()  # Add legend
plt.grid()  # Add grid

# Plot Energy Efficiency vs number of active APs (L1)
plt.figure()
plt.plot(L - np.arange(0,L), LB_EE.transpose(), 'b.', label='Lower Bound')
plt.plot(L - np.arange(0,L), Avg_EE.transpose(), 'g.', label='Median (Avg)')
plt.plot(L - np.arange(0,L), UB_EE.transpose(), 'r.', label='Upper Bound')
plt.axis([1, 87, 10, 45])  # Set axis limits
plt.ylabel("EE [Mbit/Joule]")  # Y-axis label
plt.xlabel("L1")  # X-axis label (number of active APs)
plt.legend()  # Add legend
plt.grid()  # Add grid