# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:02:24 2019

@author: user
"""

#%Analystical BER vs SNR
# BPSK
globalReps = 1000
globalError = np.empty([globalReps, len(SNR)])
p = 0.5*erfc(np.sqrt(R*Eb_No_lin))
codederrorBPSK = 1-(1-p)**n - n*p*(1-p)**(n-1)
theoreticalErrorBPSK = 0.5*erfc(np.sqrt(Eb_No_lin))
for i_global in range(globalReps):
    for i_snr in range(np.size(SNR)):
        snr = SNR[i_snr]
        
        '''
               Generate channel Input
        '''
        u = fn.generateU(N,k)
        x = fn.generteCodeWord(N, n, u, G)
        
        ''' 
            Channel Encoding: BPSK
        '''
        xflat = np.reshape(x, [-1])
        xBPSK = fn.BPSK(xflat)
        yflat = fn.AWGN(xBPSK,snr)
        ychannel = yflat.reshape(N,n) # noisy codewords
        
        '''
            Decoding
        '''
        # Hard decision (hamming distance)
        y = ychannel
        MAP = np.empty([N,k]) # decoded
        for i in range(N):
            MAP[i] = fn.hardMAPAWGN(y[i], messages, possibleCodewords)
        
        '''
            Error Calculation
        '''
        globalError[i_global][i_snr] = fn.codeErrorFunction(MAP, u)
        
'''
    MC error treatment
'''
avgHardError = np.average(globalError, 0)

'''
    Save Data
'''
filename = './Data/simu/simu-vs-theory.pickle'
with open(filename,  'wb') as f:
    pickle.dump([avgHardError, theoreticalErrorBPSK, codederrorBPSK], f)