# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 21:47:43 2019

@author: user
"""

title = 'MLNN'
directory = 'Models/'+title+'/'
layerWidth = [128,64,32,k]
lw = str(layerWidth).replace(" ", "")
elevado = 10
numEpochs=2**elevado

runfile('C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN/DNN-array-decoder.py', 
        wdir='C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN')

title = 'MLNN'
directory = 'Models/'+title+'/'
layerWidth = [128,64,32,k]
lw = str(layerWidth).replace(" ", "")
elevado = 12
numEpochs=2**elevado

runfile('C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN/DNN-array-decoder.py', 
        wdir='C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN')

title = 'MLNN'
directory = 'Models/'+title+'/'
layerWidth = [128,64,32,k]
lw = str(layerWidth).replace(" ", "")
elevado = 14
numEpochs=2**elevado

runfile('C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN/DNN-array-decoder.py', 
        wdir='C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN')

title = 'MLNN'
directory = 'Models/'+title+'/'
layerWidth = [128,64,32,k]
lw = str(layerWidth).replace(" ", "")
elevado = 18
numEpochs=2**elevado

runfile('C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN/DNN-array-decoder.py', 
        wdir='C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN')


#%%

'''
    1H Decoder batch run
'''

title = 'NN1H'
directory = 'Models/'+title+'/'
layerWidth = [256]
lw = str(layerWidth).replace(" ", "")
elevado = 10
numEpochs=2**elevado

runfile('C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN/DNN-1H-decoder.py', 
        wdir='C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN')

title = 'NN1H'
directory = 'Models/'+title+'/'
layerWidth = [256]
lw = str(layerWidth).replace(" ", "")
elevado = 12
numEpochs=2**elevado

runfile('C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN/DNN-1H-decoder.py', 
        wdir='C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN')

title = 'NN1H'
directory = 'Models/'+title+'/'
layerWidth = [256]
lw = str(layerWidth).replace(" ", "")
elevado = 14
numEpochs=2**elevado

runfile('C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN/DNN-1H-decoder.py', 
        wdir='C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN')


############################################
#%%
'''
    Array Autoencoder batch run
'''

# title = 'AAutoencoder'
# directory = 'Models/'+title+'/'
# encoderNodes = [32, 64, 128, 16]
# decoderNodes = [128, 64, 32, 8]
# lw = str(encoderNodes).replace(" ", "")+str(decoderNodes).replace(" ", "")
# elevado = 2
# numEpochs=2**elevado

# runfile('C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN/Array_Autoencoder.py', 
#         wdir='C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN')

train_snr = 1.4
title = 'AAutoencoder'
directory = 'Models/'+title+'/'
encoderNodes = [32, 64, 128, 16]
decoderNodes = [128, 64, 32, 8]
lw = str(encoderNodes).replace(" ", "")+str(decoderNodes).replace(" ", "")
elevado = 14
numEpochs=2**elevado

runfile('C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN/Array_Autoencoder.py', 
        wdir='C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN')

title = 'AAutoencoder'
directory = 'Models/'+title+'/'
encoderNodes = [32, 64, 128, 16]
decoderNodes = [128, 64, 32, 8]
lw = str(encoderNodes).replace(" ", "")+str(decoderNodes).replace(" ", "")
elevado = 16
numEpochs=2**elevado

runfile('C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN/Array_Autoencoder.py', 
        wdir='C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN')

title = 'AAutoencoder'
directory = 'Models/'+title+'/'
encoderNodes = [32, 64, 128, 16]
decoderNodes = [128, 64, 32, 8]
lw = str(encoderNodes).replace(" ", "")+str(decoderNodes).replace(" ", "")
elevado = 18
numEpochs=2**elevado

runfile('C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN/Array_Autoencoder.py', 
        wdir='C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN')

#%
# title = 'AAutoencoder'
# directory = 'Models/'+title+'/'
# encoderNodes = [256, 256, 16, 16]
# decoderNodes = [128, 64, 32, 8]
# train_snr = 1
# lw = str(encoderNodes).replace(" ", "")+str(decoderNodes).replace(" ", "")+str(train_snr)
# elevado = 18
# numEpochs=2**elevado


# runfile('C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN/Array_Autoencoder.py', 
#         wdir='C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN')

#%
# title = '1HAutoencoder'
# timestr = time.strftime("%Y%m%d-%H%M%S")
# elevado = 12
# numEpochs = 2**elevado
# batchSize = 256
# directory = 'Models/'+title+'/'
# encoderNodes = [256, 16]
# decoderNodes = [256,256]
# train_snr = 1.4
# lw = str(encoderNodes).replace(" ", "")+str(decoderNodes).replace(" ", "")+str(train_snr)



# runfile('C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN/1H_Autoencoder.py', 
#         wdir='C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN')
#%%
elevados = [10, 12, 14]
for elevado in elevados:
    title = '1HAutoencoder'
    timestr = time.strftime("%Y%m%d-%H%M%S")
    numEpochs = 2**elevado
    batchSize = 256 
    directory = 'Models/'+title+'/'
    encoderNodes = [1024, 16]
    decoderNodes = [256]
    lw = str(encoderNodes).replace(" ", "") +str(decoderNodes).replace(" ", "")+str(train_snr_db)+'categorical'
    runfile('C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN/1H_Autoencoder.py', 
            wdir='C:/Users/user/Desktop/GitHub/PIR/MyCode/AWGN')

#%%
'''
    Prediction
'''
title = 'MLNN'
layerWidth = [128,64,32,k]
elevado = 18
numEpochs=2**elevado
lw = str(layerWidth).replace(" ", "")
filename = './Models/'+title+'/MLNN_[128,64,32,8]_Mep_262144.h5'
DecoderMLNN = tf.keras.models.load_model(filename)
Decoder=DecoderMLNN.layers[1]
fn.MLNNSinglePredictionAWGN(G, Decoder, SNR, globalReps, N, n, k, lw,numEpochs, title)

'''
    Ploting
'''
filename = './Data/MLNN/MLNN_'+lw+'_Mep_'+str(numEpochs)+'.pickle'
with open(filename, 'rb') as f:
    avgMLNNError = pickle.load(f)
markerlist = ['', '', '^']
linelist = ['-', '--', '']
colorlist = ['k', 'k', 'k']
fig = fn.plotAWGN([Eb_No_dB,Eb_No_dB, Eb_No_dB], [theoreticalErrorBPSK, avgMAPError, 
             avgMLNNError ], 
            ['Uncoded BPSK', 'Soft MAP Decoder', '$M_{ep} = 2^{'+str(elevado)+'}$'],
            colorlist, linelist, markerlist,
            lineWidth, markerSize)
plt.xlim([SNRdbmin, SNRdbmax])
plt.ylim([2*10**-6, 8*10**-1])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('Results/MLNN_vs_Decoder_'+lw+'_Mep_'+str(numEpochs)+'.png', bbox_inches='tight', dpi=300)

#%%
'''
    Ploting
'''
title = 'NN1H'
layerWidth = [256]
elevado = 12
numEpochs=2**elevado
lw = str(layerWidth).replace(" ", "")
filename = './Models/'+title+'/NN1H_[256]_Mep_4096.h5'
DecoderML1HN = tf.keras.models.load_model(filename)
Decoder = DecoderML1HN.layers[1]
fn.NN1HSinglePredictionAWGN(G, Decoder, SNR, globalReps, N, n, k, lw,numEpochs, title, messages, train_snr)

filename = './Data/NN1H/NN1H_[256]_Mep_4096_1.pickle'
with open(filename, 'rb') as f:
    avgNN1HError = pickle.load(f)
markerlist = ['', '', '^']
linelist = ['-', '--', '']
colorlist = ['k', 'k', 'k']
fig = fn.plotAWGN([Eb_No_dB,Eb_No_dB,Eb_No_dB], [theoreticalErrorBPSK,
                  avgMAPError,avgNN1HError], 
            [ 'Uncoded BPSK', 'Soft MAP Decoder','$M_{ep} = 2^{12}$'],
            colorlist, linelist, markerlist,
            lineWidth, markerSize)
plt.xlim([SNRdbmin, SNRdbmax])
plt.ylim([2*10**-6, 8*10**-1])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('Results/NN1H_vs_MAP_'+lw+'_Mep_'+str(numEpochs)+'_'+str(train_snr)+'.png', bbox_inches='tight', dpi=300)


