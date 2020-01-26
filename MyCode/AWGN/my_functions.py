# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:28:03 2019

@author: Eduardo D
"""


from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc # complementary error function
import itertools
from scipy import stats
import tensorflow as tf
from tensorflow.keras import layers
from keras import backend as K
import os

import pickle

def plotit(t, u, title="Title", X="time (s)", Y="Amplitude"):
    plt.figure()
    plt.plot(t,u,'b')
    
    plt.grid(True)
    
    plt.title(title)
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.show()
    return

def scatterit(t, u, title="Title", X="time (s)", Y="Amplitude"):
    plt.figure()
    plt.plot(t,u, 'o')
    
    plt.grid(True)
    
    plt.title(title)
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.show()
    return

def sineWave(t, mean, amplitude, f):
    omega = 2 * pi * f # fs > 2f
    return amplitude * sin(omega * t + 0) + mean

def analog2binary(t, bCodification, inputSignal):
    # Example: we divide the amplitude in a scale of 8 bits and each sequence of 8 bits represents a point
    # in the end we should have t/T * 8 bits
    u = inputSignal
    M = len(u)
    x = 2**bCodification * (
    u - mean(u) - min(u - mean(u))) / max(u - mean(u) - min(u - mean(u)))
    
    x = x.astype(int)
    
    scatterit(t, x, "Quantized signal")
    b = ''  # strings of all bits togheter
    for i in range(M):
        str = "{0:b}".format(x[i])  # variable length source code word
        str = str.zfill(bCodification)  
        b = b + str  # transform b in an array of symbols 0's and 1's
    
    return b
    
def PSK2 (b, V):
    # 2 - PSK: 2 different phase representations. Maps binary symbols unsing the map 0 -> -1 and 1-> 1. Constellation of size 2
    um = empty(len(b))  # u - modulated (symbols modulated)
    for i in range(len(b)): # step function
        if (b[i] == '0'):
            um[i] = -1*V
        else:
            um[i] = 1*V
            
    return um

def pulseSignal(A, d): 
    # d is the discretization of the pulse
    pulse1 = A*ones(int(d/2))
    pulse2 = zeros(int(d/2))
    
    return concatenate((pulse1, pulse2), axis = None)
    
def constantSignal(A, d): 
    # d is the discretization of the pulse
    pulse1 = A*ones(int(d))
    return pulse1
    
    
def pulses2waveform (N,d,p,uk):
    uwv = empty(N*d)
    for i in range(len(uwv)):
        uwv[i] = uk[int(i/d)]*p[i%d]
    return uwv
    
def fourriertransform(x, interval, fc, BB):
    freq = fft.fftfreq(len(x), interval) # Frequency space
    xhat = fft.fft(x)
    fig, ax = plt.subplots()

    ax.plot(freq, xhat.real)
    ax.set_title('Waveform Fourrier Tranform')
    ax.set_xlabel('Frequency in Hertz [Hz]')
    ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
    ax.set_xlim(-fc-BB, fc+BB)
    
    return concatenate((freq, xhat), axis=0)
    
    
def Q(x):
    return 0.5*erfc(x/sqrt(2))

def MPSK_BER(M,Eb,No):
    n = log2(8)
    Es = n*Eb # energy per symbol: nEb where 2^n = M
    gamma_s = Es/No
    return 2*Q(sqrt(2*gamma_s)*sin(pi/M))/n

def empiricBER(u,uhat):
    n = len(u)
    indicatrice = u!=uhat
    P = sum(indicatrice)/n
    return P

def binarySignalPower(u):
    n = len(u)
    return sum(u**2)/n

def BSC(b, p):
    decision = random.rand(len(b))
    noise = decision < p      
    return (noise+b)%2
    
def bitEnergy(Eb, N0):
    return Eb/N0
        
def Hb(p):
    return -p*log2(p)-(1-p)*log2(1-p)

def matrixGenerator(H, name):
    n = name[0]
    k = name[1]
    Ik = eye(k)
    P = H[:,:n-(n-k)].T
    G = concatenate((Ik, P), axis = 1)
    return G 

def parityMatrix(name): # arbitrary rule
    n = name[0]
    k = name[1]
    n_p = n-k # number of parity bits
    limit = k-n_p+1
    e = eye(k) # message generator
    P = empty([k,n-k])
    for l in range(0,k):
        b = e[l]
        p = empty(n_p)
        for j in range(0,n_p):
            aux = b[j:limit+j]
            p[j] = sum(aux)%2
        P[l] = p
    return P

def parityCheckMatrix(name):
    n = name[0]
    k = name[1]
    m = n-k
    tuples = asarray(list(itertools.product(*[(0, 1)] * m)))
    H = tuples[1:].T
    soma = sum(H,0)
    counter = size(H,1)
    i = 0
    while i < counter:
        if(soma[i]<=1):
            H = delete(H,i,1)
            counter-=1
            i-=1
            soma = sum(H,0)
        i+=1
    return concatenate((H,eye(m)),axis=1)
    
        
def parityCheck(G):
    k = size(G, 0)
    n = size(G,1)
    I = eye(n-k)
    P = G[:k,k:]
   
    H = concatenate((-P.T%2,I), axis = 1)
    C = dot(G,H.T)
    return print('Parity check: \n', sum(C))

def hammingDistance(a,b):
    return np.sum(a!=b, axis = 1)

def possibleCodewordsH(name):
    n = name[0]
    k = name[1]
    H = parityCheckMatrix(name)
    tuples = asarray(list(itertools.product(*[(0, 1)] * n)))
    
    counter = size(tuples,0)
    i = 0
    while i < counter:
        if(sum(dot(tuples[i],H.T)%2)!=0):
            tuples = delete(tuples,i,0)
            counter-=1
            i-=1
        i+=1
    return tuples

def possibleCodewordsG(name, G):
    n = name[0]
    k = name[1]
    tuples = asarray(list(itertools.product(*[(0, 1)] * k)))
    words = empty([size(tuples,0), n])
    for i in range(size(tuples,0)):
        words[i] = dot(tuples[i], G)%2
    return tuples, words

def syndrome(y, H, name):
    n = name[0]
    k = name[1]
    N = size(y,0)
    S = empty([N,n-k]) # identify if there are errors - syndrome
    # k stored syndromes
    for i in range(N):
        S[i] = dot(y[i],H.T)%2 # syndrome row vector
    return S
            
    
def codeErrorFunction(y, x):
    Ecw = sum(y != x) # Codeword Error with hamming decoding
    return Ecw/size(x)

def bitErrorFunction(uhat, u):
    Eb = sum(uhat != u)
    return Eb/size(u)
    
def generateU(N,k):
    return stats.bernoulli.rvs(0.5,size=[N,k]) # input message matrix

def generteCodeWord(N, n, u, G):
    x = empty([N,n]) # code words
    for i in range(N):
        x[i] = dot(u[i],G)%2 # codeword row vector         
    return x

###################
# AWGN
###################
def BPSK(x):
    return x*2-1

def decodeAWGN(x):
    return (np.sign(x) == 1).astype(float)

def hardMAPAWGN(x, messages, possibleCodewords):
    y = (np.sign(x) == 1).astype(float)
    minDistWord = np.argmin(hammingDistance(possibleCodewords, y), 0) # find word of minimum distance
    MAP = messages[minDistWord]
    return MAP

def h_norm_np(x):
    return (x-np.mean(x))/np.sqrt(np.var(x))

def linear2dB(x):
    return 10*np.log(x)

def AWGN(x, snr):
    sigma = np.sqrt(1/(2*snr)) # scaling factor
    n = np.size(x)
    noise = np.random.normal(0, sigma, n)
    return x + noise

def euclidianDistance(possibleCodewords, y):
    return np.linalg.norm(possibleCodewords-y, ord=2, axis=1)
    
def plotAWGN(xlist, ylist, legend, colorlist, linelist, markerlist, lineWidth, markerSize, X="Eb/No (dB)", Y="BER"):
    fig = plt.figure(figsize=(8, 6))
    
    for i,array in enumerate(xlist):
        plt.plot(xlist[i],ylist[i], color=colorlist[i], linewidth=lineWidth,
             linestyle=linelist[i], marker=markerlist[i], markersize=markerSize)
    
    plt.grid(True, which="both")
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.yscale('log')
    plt.legend(legend)
    return fig

def scatterAWGN(xlist, ylist, legend, colorlist, linelist, markerlist, lineWidth, markerSize, X="Eb/No (dB)", Y="BER"):
    fig = plt.figure(figsize=(8, 6))
    for j in range(2):
        plt.plot(xlist[j],ylist[j], color=colorlist[j], linewidth=lineWidth,
             linestyle=linelist[j], marker=markerlist[j], markersize=markerSize)
    for i in range(2,len(xlist)):
        plt.scatter(xlist[i],ylist[i], color=colorlist[i],
             marker=markerlist[i], s=markerSize, zorder=2)
    
    plt.grid(True, which="both")
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.yscale('log')
    plt.legend(legend)
    return fig

def MLNNSinglePredictionAWGN(G, Decoder, SNR, globalReps, N, n, k, lw,numEpochs, title):
    DecoderError = np.empty([globalReps, len(SNR)])
    for i_global in range(globalReps):
        for i_snr in range(np.size(SNR)):
            snr = SNR[i_snr]
            u = generateU(N,k)
            x = generteCodeWord(N, n, u, G)
            xflat = np.reshape(x, [-1])
            xBPSK = BPSK(xflat)
            yflat = AWGN(xBPSK,snr)
            ychannel = yflat.reshape(N,n) # noisy codewords
            y = ychannel
            prediction = Decoder.predict(y)
            predictedMessages = np.round(prediction)
            DecoderError[i_global][i_snr] = bitErrorFunction(predictedMessages, u)

    avgDecoderError = np.average(DecoderError, 0)
    filename = './Data/'+ title+'/'+ title+'_'+lw+'_Mep_'+str(numEpochs)+'.pickle'
    with open(filename,  'wb') as f:
        pickle.dump(avgDecoderError, f)
    with open(filename, 'rb') as f:
        avgMLNNError = pickle.load(f)
        
        
def NN1HSinglePredictionAWGN(G, Decoder, SNR, globalReps, N, n, k, lw,numEpochs, title, messages, train_snr):
    DecoderError = np.empty([globalReps, len(SNR)])
    for i_global in range(globalReps):
        for i_snr in range(np.size(SNR)):
            snr = SNR[i_snr]
            u = generateU(N,k)
            x = generteCodeWord(N, n, u, G)
            xflat = np.reshape(x, [-1])
            xBPSK = BPSK(xflat)
            yflat = AWGN(xBPSK,snr)
            ychannel = yflat.reshape(N,n) # noisy codewords
            y = ychannel
            prediction = Decoder.predict(y)
            predictedMessages = multipleOneshot2messages(prediction, messages)
            DecoderError[i_global][i_snr] = bitErrorFunction(predictedMessages, u)

    avgDecoderError = np.average(DecoderError, 0)
    filename = './Data/'+ title+'/'+ title+'_'+lw+'_Mep_'+str(numEpochs)+'_'+str(train_snr)+'.pickle'
    with open(filename,  'wb') as f:
        pickle.dump(avgDecoderError, f)
    with open(filename, 'rb') as f:
        avgNN1HError = pickle.load(f)
        
def AutoencoderSinglePredictionAWGN(Encoder, Decoder, SNR, globalReps, N, n, k, lw,numEpochs, title):
    globalErrorAutoencoder = np.empty([globalReps, len(SNR)])
    for i_global in range(globalReps):
        for i_snr in range(np.size(SNR)):
            snr = SNR[i_snr]
            u = generateU(N,k)
            x = Encoder.predict(u)
            xflat = np.reshape(x, [-1])
            xBPSK = xflat
            yflat = AWGN(xBPSK,snr)
            ychannel = yflat.reshape(N,n)
            y = ychannel # noisy codewords
            prediction = Decoder.predict(y)
            predictedMessages = np.round(prediction)
            globalErrorAutoencoder[i_global][i_snr] = bitErrorFunction(predictedMessages, u)
            
    avgAutoencoderError = np.average(globalErrorAutoencoder, 0)
    filename = './Data/'+ title+'/'+ title+'_'+lw+'_Mep_'+str(numEpochs)+'.pickle'
    with open(filename,  'wb') as f:
        pickle.dump(avgAutoencoderError, f)
    with open(filename, 'rb') as f:
        avgAutoencoderError = pickle.load(f)
        
def Autoencoder1HSinglePredictionAWGN(Encoder, Decoder, SNR, globalReps, N, n, k, lw,numEpochs, title, messages, train_snr):
    globalErrorAutoencoder = np.empty([globalReps, len(SNR)])
    for i_global in range(globalReps):
        for i_snr in range(np.size(SNR)):
            snr = SNR[i_snr]
            u = generateU(N,k)
            u2 = u
            x = Encoder.predict(u2)
            #x = np.round(x)
            #x = BPSK(x)
            x = h_norm_np(x)
            xflat = np.reshape(x, [-1])
            xBPSK = xflat
            yflat = AWGN(xBPSK,snr)
            ychannel = yflat.reshape(N,n)
            y = ychannel # noisy codewords
            prediction = Decoder.predict(y)
            predictedMessages = multipleOneshot2messages(prediction, messages)
            globalErrorAutoencoder[i_global][i_snr] = bitErrorFunction(predictedMessages, u)
            
    avgAutoencoderError = np.average(globalErrorAutoencoder, 0)
    filename = './Data/'+ title+'/'+ title+'_'+lw+'_Mep_'+str(numEpochs)+'_'+str(train_snr)+'.pickle'
    with open(filename,  'wb') as f:
        pickle.dump(avgAutoencoderError, f)
    with open(filename, 'rb') as f:
        avgAutoencoderError = pickle.load(f)
        
def AEncoderSinglePredictionAWGN(G, Encoder, SNR, globalReps, N, n, k, lw,numEpochs, title, possibleRealCodewords, messages, train_snr):
    DecoderError = np.empty([globalReps, len(SNR)])
    for i_global in range(globalReps):
        for i_snr in range(np.size(SNR)):
            snr = SNR[i_snr]
            x = generateU(N,k)
            u = Encoder.predict(x)
            u = np.round(u)
            uflat = np.reshape(u, [-1])
            uBPSK = BPSK(uflat)
            
            yflat = AWGN(uBPSK,snr)
            ychannel = yflat.reshape(N,n) # noisy codewords
            y = ychannel
            
            #MAP
            MAP = np.empty([N,k]) # decoded
            for i in range(N):
                minDistWord = np.argmin(euclidianDistance(possibleRealCodewords, y[i]), 0) # find word of minimum distance
                MAP[i] = messages[minDistWord]
            
            DecoderError[i_global][i_snr] = codeErrorFunction(MAP, x)

    avgDecoderError = np.average(DecoderError, 0)
    filename = './Data/'+ title+'/'+ title+'_'+lw+'_Mep_'+str(numEpochs)+'_'+str(train_snr)+'.pickle'
    with open(filename,  'wb') as f:
        pickle.dump(avgDecoderError, f)

###################
# DNNs
###################
'''
    NN Functions
'''
def tensorAWGN(x):
    train_snr = 1 # Article: Gruber - polar codes 
    #snr = K.constant(train_snr,dtype=tf.float32)
    noise = K.random_normal(shape=K.shape(x),mean=0.,stddev=np.sqrt(1/(2*train_snr)))
    #noise = K.random_normal_variable(shape=(func_output_shape(x),), mean=0, scale=K.sqrt(1/(2*snr)))
    #noiseFloat = K.cast(noise, dtype=tf.float32)
    result = tf.math.add(noise, x)
    return result

def BPSK(x):
    return x*2-1

def channel_layer(x, sigma):
    w = K.random_normal(K.shape(x), mean=0.0, stddev=sigma)
    return x + w
    
def normalize(x): # |x_i| <= 1
    xmin = K.min(x)
    xmax = K.max(x)
    a = K.constant(-1)
    b = K.constant(1)
    return a + ((x-xmin)*(b-a))/(xmax-xmin)   
    
def h_norm(x):
    return (x-K.mean(x))/K.sqrt(K.var(x))
    # x1 = tf.cast(x, tf.float64)
    # encoder_dimension = func_output_shape(x)
    # print(encoder_dimension)
    # #normalization_factor = tf.reciprocal(tf.math.sqrt(tf.reduce_sum(tf.square(x1), 1))) * tf.math.sqrt(encoder_dimension)
    # normalization_factor = tf.reciprocal(K.sqrt(K.sum(K.square(x), 1)))*np.sqrt(encoder_dimension)
    
    # h_norm =  tf.transpose(tf.multiply(tf.transpose(x), normalization_factor))
    return h_norm

def func_output_shape(x):
    shape = x.get_shape().as_list()[1]
    return shape

def metricBER(y_true, y_pred):
    return K.mean(K.not_equal(y_true,y_pred))

def ber(y_true, y_pred):
    return  K.mean(K.cast(K.not_equal(y_true, K.round(y_pred)),dtype='float32'))

def metricBER1H(y_true, y_pred):
    #return K.mean(K.not_equal(K.argmax(y_true),K.argmax(y_pred)))
    return K.mean(K.not_equal(y_true,K.round(y_pred))) #????

def tensorPossibleMessages(name):
    n = name[0]
    k = name[1]
    nptuples = asarray(list(itertools.product(*[(0, 1)] * k)))
    ttuples = K.variable(nptuples)
    return ttuples

'''
    Plot training curve
'''
def plotTraining(history):
    #todo
    return

'''
    One hot message encoding
'''
def messages2onehot(u):
    n = u.shape[0]
    k = u.shape[1]
    N = 2**k
    index=np.zeros(N)
    encoded = np.zeros([n,N])
    for j in range(n):
        for i in range(k-1, -1, -1):
            index[j] = index[j] + u[j][i]*2**(k-1-i)
        encoded[j][int(index[j])] = 1
    return encoded

def singleMessage2onehot(m):
    k = m.shape[0]
    n = 2**k
    encoded = np.zeros(n)
    index = 0
    for i in range(k-1, -1, -1):
        index = index + m[i]*2**(k-1-i)
    encoded[int(index)] = 1
    return encoded

def onehot2singleMessage(h, messages):
    index = np.argmax(h)
    return messages[index]
    '''n = h.shape[0]
    k = int(np.log2(n))
    return np.asarray([int(x) for x in list(('{0:0'+str(k)+'b}').format(index))])
'''
def multipleOneshot2messages(h, messages):
    indexes = np.argmax(h,1)
    n = h.shape[1]
    k = int(np.log2(n))
    N = len(indexes)
    out = np.zeros([N, k])
    for i in range(N):
        out[i] = messages[indexes[i]]
    return out
    
def TensorOnehot2singleMessage(h):
    #todo
    index = tf.argmax(h)
    return np.asarray([int(x) for x in list('{0:08b}'.format(index))])

def roundCode(x):
    return tf.stop_gradient(K.round(x)-x)+x

def signCode(x):
    return K.sign(x)

def createDir(path):
    try:  
        os.mkdir(path)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s " % path)
        
def messages2customEncoding(messages, Encoder):
    return Encoder.predict(messages)

def arrayDecoderPrediction(G, Decoder, pOptions, globalReps, N, n, k):
    globalErrorDecoder = np.empty([globalReps, len(pOptions)])
    for i_global in range(globalReps):
        for i_p in range(np.size(pOptions)):
            p = pOptions[i_p]
            u = generateU(N,k)
            x = generteCodeWord(N, n, u, G)
            xflat = np.reshape(x, [-1])
            yflat = BSC(xflat,p)
            y = yflat.reshape(N,n) # noisy codewords
            prediction = Decoder.predict(y)
            predictedMessages = np.round(prediction)
            globalErrorDecoder[i_global][i_p] = bitErrorFunction(predictedMessages, u)
            
    return globalErrorDecoder

def onehotDecoderPrediction(G, Decoder, pOptions, globalReps, N, n, k, messages):
    globalErrorDecoder = np.empty([globalReps, len(pOptions)])
    for i_global in range(globalReps):
        for i_p in range(np.size(pOptions)):
            p = pOptions[i_p]
            u = generateU(N,k)
            x = generteCodeWord(N, n, u, G)
            xflat = np.reshape(x, [-1])
            yflat = BSC(xflat,p)
            y = yflat.reshape(N,n) # noisy codewords
            prediction = Decoder.predict(y)
            predictedMessages = multipleOneshot2messages(prediction, messages)
            globalErrorDecoder[i_global][i_p] = bitErrorFunction(predictedMessages, u)
            
    return globalErrorDecoder

def arrayAutoencoderPrediction(Encoder, Decoder, pOptions, globalReps, N, n, k):
    globalErrorAutoencoder = np.empty([globalReps, len(pOptions)])
    for i_global in range(globalReps):
        for i_p in range(np.size(pOptions)):
            p = pOptions[i_p]
            u = generateU(N,k)
            x = Encoder.predict(u)
            xflat = np.reshape(x, [-1])
            yflat = BSC(xflat,p)
            y = yflat.reshape(N,n) # noisy codewords
            prediction = Decoder.predict(y)
            predictedMessages = np.round(prediction)
            globalErrorAutoencoder[i_global][i_p] = bitErrorFunction(predictedMessages, u)
            
    return globalErrorAutoencoder

def onehotAutoencoderPrediction(Encoder, Decoder, messages, pOptions, globalReps, N, n, k):
    globalErrorAutoencoder1H = np.empty([globalReps, len(pOptions)])
    for i_global in range(globalReps):
        for i_p in range(np.size(pOptions)):
            p = pOptions[i_p]
            u = generateU(N,k)
            u1h = messages2onehot(u)
            x = np.round(Encoder.predict(u1h))
            xflat = np.reshape(x, [-1])
            yflat = BSC(xflat,p)
            y = yflat.reshape(N,n) # noisy codewords
            prediction = Decoder.predict(y)
            predictedMessages = multipleOneshot2messages(prediction, messages)
    
            globalErrorAutoencoder1H[i_global][i_p] = bitErrorFunction(predictedMessages, u)
    
    return globalErrorAutoencoder1H

def clipp(x):
    return K.clip(x,-1,1)