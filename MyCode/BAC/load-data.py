# -*- coding: utf-8 -*-
'''
    Uncoded simulation
'''
filename = './Data/simu/simu-vs-theory.pickle'
with open(filename, 'rb') as f:
    avgHardError, theoreticalErrorBPSK, uncodederrorBPSK = pickle.load(f)
#avgGlobalError = k/n*avgGlobalError
#theoreticalErrorBPSK = k/n*theoreticalErrorBPSK
'''
    MAP Avg Error
'''
filename = './Data/MAP/MAP.pickle'
with open(filename, 'rb') as f:
    avgMAPError = pickle.load(f)
    
'''
    MLNN Decoder
'''
filename = './Data/MLNN/MLNN_[128,64,32,8]_Mep_4096_.pickle'
with open(filename, 'rb') as f:
    avgMLNNError = pickle.load(f)