# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 14:58:34 2019

@author: user
"""
#%%
from timeit import default_timer as timer

timeBPSK = np.array([])
timeMAP = np.array([])
timeMLNN = np.array([])
timeNN1H = np.array([])
#timeAA = np.array([])
time1H = np.array([])
#timeAEncoder = np.array([])

title = 'MLNN'
# layerwidth = [128,64,32,k]
# elevado = 18
# numepochs=2**elevado
# lw = str(layerwidth).replace(" ", "")
filename = './models/'+title+'/mlnn_[128,64,32,8]_mep_262144.h5'
decoder = tf.keras.models.load_model(filename)
DecoderMLNN = decoder.layers[1]

title = 'NN1H'
filename = './Models/'+title+'/NN1H_[256]_Mep_4096.h5'
Decoder = tf.keras.models.load_model(filename)
DecoderNN1H = Decoder.layers[1]

title = '1HAutoencoder'
filename = './Models/'+title+'/1HAutoencoder_[1024,16][256]1categorical_Mep_16384_1.2589254117941673.h5'
Autoencoder1H = tf.keras.models.load_model(filename)
Encoder1H = Autoencoder1H.layers[0]
Decoder1H = Autoencoder1H.layers[2]

#%%
for i_global in range(100):
    print(i_global)
    '''
        Common variables
    '''
    N = 100 # number of messages
    snr = 5 #dB
    u = fn.generateU(N,k)
    
    '''
        	BPSK code transmission
    '''
    start = timer()
    x = fn.generteCodeWord(N, n, u, G)
    xflat = np.reshape(x, [-1])
    xBPSK = fn.BPSK(xflat)
    yflat = fn.AWGN(xBPSK,snr)
    ychannel = yflat.reshape(N,n) # noisy codewords
    y = ychannel
    uhat = fn.decodeAWGN(y)
    end = timer()
    timeBPSK = np.append(timeBPSK, end - start)
    #print('Time BPSK: ', timeBPSK)
    
    '''
        	MAP code transmission
    '''
    start = timer()
    x = fn.generteCodeWord(N, n, u, G)
    xflat = np.reshape(x, [-1])
    xBPSK = fn.BPSK(xflat)
    yflat = fn.AWGN(xBPSK,snr)
    ychannel = yflat.reshape(N,n) # noisy codewords
    y = ychannel
    MAP = np.empty([N,k]) # decoded
    for i in range(N):
        minDistWord = np.argmin(fn.euclidianDistance(possibleRealCodewords, y[i]), 0) # find word of minimum distance
        MAP[i] = messages[minDistWord]
    uhat = MAP
    end = timer()
    timeMAP = np.append(timeMAP, end - start)
    #print('Time MAP: ', timeMAP)
    '''
        	Array decoder code transmission
    '''
    
    
    start = timer()
    x = fn.generteCodeWord(N, n, u, G)
    xflat = np.reshape(x, [-1])
    xBPSK = fn.BPSK(xflat)
    yflat = fn.AWGN(xBPSK,snr)
    ychannel = yflat.reshape(N,n) # noisy codewords
    y = ychannel
    prediction = DecoderMLNN.predict(y)
    uhat = np.round(prediction)
    end = timer()
    timeMLNN = np.append(timeMLNN, end - start)
    #print('Time MLNN: ', timeMLNN)
    '''
        	1H Array decoder code transmission
    '''
    
    
    start = timer()
    x = fn.generteCodeWord(N, n, u, G)
    xflat = np.reshape(x, [-1])
    xBPSK = fn.BPSK(xflat)
    yflat = fn.AWGN(xBPSK,snr)
    ychannel = yflat.reshape(N,n) # noisy codewords
    y = ychannel
    prediction = DecoderNN1H.predict(y)
    uhat = fn.multipleOneshot2messages(prediction, messages)
    end = timer()
    timeNN1H = np.append(timeNN1H, end - start)
    #print('Time NN1H: ', timeNN1H)
    # '''
    #     	Array Autoencoder code transmission
    # '''
    # filename
    # Autoencoder = tf.keras.models.load_model(filename)
    # Encoder = Autoencoder.layers[0]
    # Decoder = Autoencoder.layers[2]
    # start = timer()
    # x = Encoder.predict(u)
    # xflat = np.reshape(x, [-1])
    # xBPSK = fn.BPSK(xflat)
    # yflat = fn.AWGN(xBPSK,snr)
    # ychannel = yflat.reshape(N,n) # noisy codewords
    # y = ychannel
    # prediction = Decoder.predict(y)
    # uhat = np.round(prediction)
    # end = timer()
    # timeAA = np.append(timeAA, end - start)
    
    '''
        	1H Array Autoencoder code transmission
    '''
    
    start = timer()
    x = Encoder1H.predict(u)
    xflat = np.reshape(x, [-1])
    xBPSK = fn.BPSK(xflat)
    yflat = fn.AWGN(xBPSK,snr)
    ychannel = yflat.reshape(N,n) # noisy codewords
    y = ychannel
    prediction = Decoder1H.predict(y)
    uhat = fn.multipleOneshot2messages(prediction, messages)
    end = timer()
    time1H = np.append(time1H, end - start)
    #print('Time 1HAA: ', time1H)
    # '''
    #     	Array Encoder and MAP code transmission
    # '''
    # filename
    # AEncoder = tf.keras.models.load_model(filename)
    # start = timer()
    # x = Encoder1H.predict(u)
    # xflat = np.reshape(x, [-1])
    # xBPSK = fn.BPSK(xflat)
    # yflat = fn.AWGN(xBPSK,snr)
    # ychannel = yflat.reshape(N,n) # noisy codewords
    # y = ychannel
    # MAP = np.empty([N,k]) # decoded
    # for i in range(N):
    #     minDistWord = np.argmin(fn.euclidianDistance(possibleRealCodewords, y[i]), 0) # find word of minimum distance
    #     MAP[i] = messages[minDistWord]
    # uhat = MAP
    # end = timer()
    # timeAEncoder = np.append(timeMAP, end - start)

'''
        Analyzing times
'''
meantimeBPSK = np.mean(timeBPSK[1:])
stdBPSK = np.std(timeBPSK[1:])

meantimeMAP = np.mean(timeMAP[1:])
stdMAP = np.std(timeMAP[1:])

meantimeMLNN = np.mean(timeMLNN[1:])
stdMLNN = np.std(timeMLNN[1:])

meantimeNN1H = np.mean(timeNN1H[1:])
stdNN1H = np.std(timeNN1H[1:])

# meantimeAA = np.mean(timeAA)
# stdAA = np.std(timeAA)

meantime1H = np.mean(time1H[1:])
std1H = np.std(time1H[1:])

# meantimeAEncoder = np.mean(timeAEncoder)
# stdAEncoder = np.std(timeAEncoder)

from tabulate import tabulate
header = ['BPSK', 'MAP', 'MLNN', 'NN1H', '1HA']
line1 = ['Mean', meantimeBPSK, meantimeMAP, meantimeMLNN, meantimeNN1H, meantime1H]
line2 = ['Std', stdBPSK, stdMAP, stdMLNN, stdNN1H, std1H]
table = tabulate([line1, line2], headers=header)
print(table)

f = open('time_analysis_result.txt', 'w')
f.write(table)
f.close()
