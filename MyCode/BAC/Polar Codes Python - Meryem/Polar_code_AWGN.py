"""
Created on Mon Dec  9 09:41:37 2019

@author: meryem benammar
"""

import numpy as np 
import Definitions as d_f
import matplotlib.pyplot as plt 

k = 4               # Number of input bits 
N = 16              # Number of coded bits 

design_EbN0_dB = 0 
design_snr_dB = design_EbN0_dB + 10*np.log10(float(k)/float(N)) 

Arikan_polar = 1    # Choose between Arikan's polar code, or Gruber's polar

EbN0_init = 0       # Initial BER 
EbN0_final = 8      # Final BER 
EbN0_points = 10    # Number of BER points 

num_words = 10000   # Number of transmitted frames of k bits 
N_errors_mini = 100 # Minimal number of errors required before switching  
N_iter_max = 10   # Number of iterations in order to compute errors 

plot_ber = 1        # Whether to plot the BER or not 
plot_fer = 1        # Whether to plot the FER or not 

######################################################################## 
#           Frozen bits and codebook construction
######################################################################## 

# Create all possible information words (array encoding)
d = np.zeros((2**k,k),dtype=bool)
for i in range(1,2**k):
    d[i]= d_f.inc_bool(d[i-1])

# Choose either Gruber's polar code or Arikan's construction of frozen bits A 
if Arikan_polar == 0: 
    A = d_f.polar_design_awgn(N, k, design_snr_dB)
    
else : # (Arikan's frozen bits obtained from Matlab's polar code construction)
    A = d_f.polar_design_awgn_arikan(N, k, design_EbN0_dB)  
    
# Creating all possible codewords and storing them for the MAP decoder
u = np.zeros((2**k,N),dtype=bool)
u[:,A] = d

x = np.zeros((2**k,N),dtype=bool)
for i in range(0,2**k):
    x[i] = d_f.polar_transform_iter(u[i])
    
#############################################################
#                       Testing
#############################################################  
# Converting EbN0 in dB to SNR in dB 
EbN0 = np.linspace(EbN0_init , EbN0_final, EbN0_points)
SNR = EbN0 + 10*np.log10(float(k)/float(N)) 

# Initialization of frame error vector 
nb_errors_MAP_f = np.zeros(len(EbN0),dtype=int) 
nb_sequences_MAP = np.zeros(len(EbN0),dtype=int)

# Initialization of bits error vector 
nb_errors_MAP_b = np.zeros(len(EbN0),dtype=int) 
nb_bits_MAP = np.zeros(len(EbN0),dtype=int)
 
for i in range(0,len(SNR)):  
    N_errors = 0 
    N_iter = 0 
    
    sigma= np.sqrt(float(1)/2/(10**(SNR[i]/10)))
    
    while N_errors < N_errors_mini  and N_iter < N_iter_max:
        
        N_iter += 1 
        
        # Source generator 
        np.random.seed(0)
        d_test = np.random.randint(0,2,size=(num_words,k)) 
        ind_test_dec = reduce(lambda a,b: 2*a+b, np.transpose(d_test)) 
         
        # Encoder        
        u_test = np.zeros((num_words, N),dtype=bool)
        u_test[:,A] = d_test
        
        c_test = np.zeros((num_words, N),dtype=bool)
        for iii in range(0,num_words):
            c_test[iii] = d_f.polar_transform_iter(u_test[iii]) 
             
        # Modulator (BPSK)
        x_test = -2*c_test + 1
    
        # Channel (AWGN)
        y_test = x_test + sigma*np.random.standard_normal(x_test.shape) 
        
        # MAP Decoder (sequence MAP decoder, not bit-map)
        for i_test in range(0, num_words):
            
            d_test_MAP = d_test[i_test,:]
            y_test_MAP = y_test[i_test,:]
            ind_test_MAP = ind_test_dec[i_test]
            
            log_APP = np.zeros((2**k,1),dtype= float)
            for i_info in range(0, len(log_APP)):
                diff_count = np.sum((y_test_MAP - (1-2*x[i_info,:]*1))**2); 
                log_APP[i_info] = - diff_count; 
              
            # Find the right index of the codeword  
            ind_inf = np.argmax(log_APP)  
            nb_errors_MAP_f[i] += 1.0* (ind_inf != ind_test_MAP) 
            nb_sequences_MAP[i] += 1
            
            # Find the right binary codeword  
            output_seq_MAP = d[ind_inf,:]
            nb_errors_MAP_b[i] += np.sum(np.mod(output_seq_MAP*1 + d_test_MAP ,2))
            nb_bits_MAP[i]+= k 
                
            N_errors += np.sum(1.0*(output_seq_MAP != d_test_MAP))    
   
#############################################################
#                      Plots 
#############################################################  
if plot_ber == 1: 
    
    legend = [] 
    plt.figure()
    
    # BER of the MAP decoder 
    plt.plot(EbN0, np.float32(nb_errors_MAP_b)/np.float32(nb_bits_MAP),'r')
    legend.append('MAP')
    
    # Parameters of the plot 
    plt.legend(legend, loc=3)
    plt.yscale('log')
    plt.xlabel('$E_b/N_0$')
    plt.ylabel('BER')
    plt.title('BER for the MAP decoder')    
    plt.grid(True)
    plt.show()
 
if plot_fer == 1: 
    
    legend = [] 
    plt.figure()
    
    # BER of the MAP decoder 
    plt.plot(EbN0, np.float32(nb_errors_MAP_f)/np.float32(nb_sequences_MAP))
    legend.append('MAP')
    
    # Parameters of the plot 
    plt.legend(legend, loc=3)
    plt.yscale('log')
    plt.xlabel('$E_b/N_0$')
    plt.ylabel('FER')    
    plt.title('FER for the MAP decoder')  
    plt.grid(True)
    plt.show()