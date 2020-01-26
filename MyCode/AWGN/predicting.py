# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 09:44:39 2019

@author: user
"""
#%%
title = '1HAutoencoder'
encoderNodes = [256, 16]
decoderNodes = [256,256]
train_snr = 1.4
lw = str(encoderNodes).replace(" ", "")+str(decoderNodes).replace(" ", "")+str(train_snr)
numEpochslist = [2**12, 2**14, 2**16, 2**18]
directory = 'Models/'+title+'/'
modelnames = []
for i in range(4):
    '''
        Load model
    '''
    numEpochs = numEpochslist[i]
    filename = directory+title+'_'+lw+'_Mep_'+str(numEpochs)+'_'+str(train_snr)+'.h5'
    Autoencoder = keras.models.load_model(filename)
    #Autoencoder = keras.models.load_model(directory+'1HAutoencoder_[256,16][256,256]14_Mep_4096_14.h5')
    Encoder = Autoencoder.layers[0]
    Decoder = Autoencoder.layers[2]
    '''
        Predict
    '''
    
    fn.Autoencoder1HSinglePredictionAWGN(Encoder, Decoder, SNR, globalReps, N, n, k, lw,numEpochs, title, messages, train_snr)
    
    '''
        Ploting
    '''
    filename = './Data/'+ title+'/'+ title+'_'+lw+'_Mep_'+str(numEpochs)+'_'+str(train_snr)+'.pickle'
    with open(filename, 'rb') as f:
        avg1HAutoencoderError = pickle.load(f)
    markerlist = ['','', 'o']
    linelist = ['-','--', '-']
    colorlist = ['k', 'k', 'k']
    fig = fn.scatterAWGN([Eb_No_dB,Eb_No_dB,Eb_No_dB], 
                         [theoreticalErrorBPSK, avgMAPError, 
                 avg1HAutoencoderError], 
                ['Uncoded BPSK', 'Soft MAP Decoder', 'One-hot Autoencoder'],
                colorlist, linelist, markerlist,
                lineWidth, markerSize)
    plt.xlim([SNRdbmin, SNRdbmax])
    plt.ylim([2*10**-6, 10**-1])
    plt.show()
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    fig.set_size_inches(width, height)
    fig.savefig('./Results/1HAutoencoder_vs_MAP_'+lw+'_Mep_'+str(numEpochs)+'_'+str(train_snr)+'.png', bbox_inches='tight', dpi=300)

#%%
title = '1HAutoencoder'
encoderNodes = [1024, 16]
decoderNodes = [1024,256]
train_snr = 1.4
lw = str(encoderNodes).replace(" ", "")+str(decoderNodes).replace(" ", "")+str(train_snr)
numEpochslist = [2**12, 2**14, 2**16]
directory = 'Models/'+title+'/'
modelnames = []
for i in range(len(numEpochslist)):
    '''
        Load model
    '''
    numEpochs = numEpochslist[i]
    filename = directory+title+'_'+lw+'_Mep_'+str(numEpochs)+'_'+str(train_snr)+'.h5'
    Autoencoder = keras.models.load_model(filename)
    #Autoencoder = keras.models.load_model(directory+'1HAutoencoder_[256,16][256,256]14_Mep_4096_14.h5')
    Encoder = Autoencoder.layers[0]
    Decoder = Autoencoder.layers[2]
    '''
        Predict
    '''
    
    fn.Autoencoder1HSinglePredictionAWGN(Encoder, Decoder, SNR, globalReps, N, n, k, lw,numEpochs, title, messages, train_snr)
    
    '''
        Ploting
    '''
    filename = './Data/'+ title+'/'+ title+'_'+lw+'_Mep_'+str(numEpochs)+'_'+str(train_snr)+'.pickle'
    with open(filename, 'rb') as f:
        avg1HAutoencoderError = pickle.load(f)
    markerlist = ['','', 'o']
    linelist = ['-','--', '-']
    colorlist = ['k', 'k', 'k']
    fig = fn.scatterAWGN([Eb_No_dB,Eb_No_dB,Eb_No_dB], 
                         [theoreticalErrorBPSK, avgMAPError, 
                 avg1HAutoencoderError], 
                ['Uncoded BPSK', 'Soft MAP Decoder', 'One-hot Autoencoder'],
                colorlist, linelist, markerlist,
                lineWidth, markerSize)
    plt.xlim([SNRdbmin, SNRdbmax])
    plt.ylim([2*10**-6, 10**-1])
    plt.show()
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    fig.set_size_inches(width, height)
    fig.savefig('./Results/1HAutoencoder_vs_MAP_'+lw+'_Mep_'+str(numEpochs)+'_'+str(train_snr)+'.png', bbox_inches='tight', dpi=300)
   
