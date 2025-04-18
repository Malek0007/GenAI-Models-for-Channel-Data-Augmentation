import math as m
import numpy as np
import matplotlib.pyplot as plt
import SignalTools
import pdb


M = 16
modulation_type = "qam" # qam or psk or pam

b_M = m.log2(M)
symbs_number = 10000

bits_number = int(b_M*symbs_number)

bits = SignalTools.binary_stream_generator(bits_number, 3498)

mod_symbs = SignalTools.mapping(bits, M, modulation_type)

mapped, bits_mapped = SignalTools.get_symbol_mapping(M, modulation_type)



snr = 20 # in dB
[noisy_symbs, pn, ps] = SignalTools.awgn_channel_generator(mod_symbs, snr, modulation_type)



llrs = SignalTools.compute_llrs("maxlog", M, modulation_type, noisy_symbs, snr)

## Soft Demapping
bit_est = SignalTools.get_bits_from_llrs(llrs, bits_number)
errors = np.sum(bit_est != bits)
ber_soft = errors / bits_number
print("BER soft : " + str(ber_soft))

## Hard Demapping
bits_hard = SignalTools.hard_demapping(noisy_symbs, modulation_type, M)
errors_hard = np.sum(bits_hard != bits)
ber_hard = errors_hard / bits_number
print("BER hard : " + str(ber_soft))



## Plot constellation
plt.figure()
plt.plot(np.real(noisy_symbs[:5000]), np.imag(noisy_symbs[:5000]), 'ro', markersize=2)
plt.plot(np.real(mod_symbs[:5000]), np.imag(mod_symbs[:5000]), 'b*', markersize=2)
plt.xlabel("Q")
plt.ylabel("I")
plt.grid("on")
plt.legend(["With noise, SNR="+str(snr)+"dB", "Without noise"], loc='upper right')
plt.show()
