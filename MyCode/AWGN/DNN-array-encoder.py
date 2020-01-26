# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 15:37:52 2019

@author: user
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:01:59 2019

@author: user
"""
'''
    Array Decoder
'''
# Training parameters
title = 'AEncoder'
timestr = time.strftime("%Y%m%d-%H%M%S")
elevado = 10
numEpochs = 2**elevado
batchSize = 256

u_train_labels = possibleCodewords.copy()
x_train_data = messages.copy()
trainSize = np.size(x_train_data, 0)

u_train_labels = np.repeat(u_train_labels, 10, axis=0)
x_train_data = np.repeat(x_train_data, 10, axis=0)
trainSize = np.size(x_train_data, 0)

train_snr_db = 1
train_snr_lin = 10**(train_snr_db/10)
train_sigma=np.sqrt(1/(2*train_snr_lin))

'''
    Sequential Model: most simple tf AEncoder model
'''
layerWidth = [128,64,n]
lw = str(layerWidth).replace(" ", "")
train_snr = 1
AEncoderEncoder = tf.keras.Sequential([ # Array to define layers
              # Adds a densely-connected layer with n units to the model: L1
              layers.Dense(layerWidth[0], activation='relu', input_shape=(k,), name='HL1'),
              # Add another: L2
              layers.Dense(layerWidth[1], activation='relu', name='HL2'),
              # Add another: L3
              #layers.Dense(layerWidth[2], activation='relu', name='HL3'),
              # Add layer with k output units:
              layers.Dense(layerWidth[2], activation='sigmoid', name='Output'),
              #layers.Lambda(fn.roundCode, output_shape=(n,))
              ], name='Array_Encoder')
# NoiseL = tf.keras.Sequential([
#         layers.Lambda(fn.roundCode, output_shape=(n,)),
#         # Noise Layer
#         layers.Lambda(fn.channel_layer, arguments={'sigma': train_sigma},input_shape=(n,), output_shape=(n,), name='Noise'),
#         #layers.GaussianNoise(stddev=np.sqrt(1/(2*train_snr)), input_shape=(n,))
#         ], name='Noise')

AEncoder = tf.keras.Sequential([AEncoderEncoder])
plot_model(AEncoder,to_file='GraphNN/'+title+'/'+title+'_'+lw+'_'+timestr+'.pdf',show_shapes=True)
    
'''
    Overall Settings/ Compilation
'''
lossFunc = 'binary_crossentropy'
AEncoder.compile(loss=lossFunc ,
              optimizer='adam')
'''
    Summaries and checkpoints 
'''
summary = AEncoder.summary()
callbacks_list = []
''' 
    Training
'''
history = AEncoder.fit(x_train_data, u_train_labels, epochs=numEpochs, 
                   batch_size=batchSize, shuffle=True, verbose=0, callbacks=callbacks_list)

# summarize history for loss
trainingFig = plt.figure(figsize=(8, 6), dpi=80)
plt.title('Batch size = '+str(batchSize))
plt.plot(history.history['loss']) # all outputs: ['acc', 'loss', 'val_acc', 'val_loss']
#plt.plot(history.history['metricBER'])
plt.grid(True, which='both')
#plt.plot(history.history['val_loss'])
plt.xlabel('$M_{ep}$')
plt.xscale('log')
plt.legend([lossFunc + ' loss'])
plt.show()

trainingFig.savefig('GraphNN/'+title+'/'+title+'_train'+ timestr+'.png', bbox_inches='tight', dpi=300)

'''
    Saving model
'''
AEncoder.save('Models/'+title+'/'+title+'_'+lw+'_Mep_'+str(numEpochs)+'.h5')  # creates a HDF5 file


'''
    Prediction
'''
Encoder=AEncoder
fn.AEncoderSinglePredictionAWGN(G, Encoder, SNR, globalReps, N, n, k, lw,numEpochs, title, possibleRealCodewords, messages, train_snr)

'''
    Ploting
'''
filename = './Data/'+ title+'/'+ title+'_'+lw+'_Mep_'+str(numEpochs)+'_'+str(train_snr)+'.pickle'
with open(filename, 'rb') as f:
    avgEncoderError = pickle.load(f)
markerlist = ['','', 'o']
linelist = ['-','--', '-']
colorlist = ['k', 'k', 'k']
fig = fn.scatterAWGN([Eb_No_dB,Eb_No_dB,Eb_No_dB], 
                     [avgGlobalError, avgMAPError, 
             avgEncoderError], 
            ['BPSK (16,8)', 'Soft MAP Decoder', 'DNN Encoder + MAP'],
            colorlist, linelist, markerlist,
            lineWidth, markerSize)
plt.xlim([SNRdbmin, SNRdbmax])
plt.ylim([2*10**-6, 10**-1])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('Results/AEncoder_vs_MAP_'+lw+'_Mep_'+str(numEpochs)+'_'+str(train_snr)+'.png', bbox_inches='tight', dpi=300)
