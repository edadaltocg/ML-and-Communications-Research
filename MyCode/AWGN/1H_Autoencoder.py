# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 14:02:33 2019

@author: user
"""
'''
    Array Autoencoder
'''
# Training parameters
# title = '1HAutoencoder'
# timestr = time.strftime("%Y%m%d-%H%M%S")
# elevado = 12
# numEpochs = 2**elevado
# batchSize = 256

u_train_labels = fn.messages2onehot(messages.copy())
x_train_data = messages.copy()

u_train_labels = np.repeat(u_train_labels, 10, axis=0)
x_train_data = np.repeat(x_train_data, 10, axis=0)
trainSize = np.size(x_train_data, 0)

train_snr_db = 1
train_snr = 10**(train_snr_db/10)
train_sigma=np.sqrt(1/(2*train_snr))
'''
    Sequential Model: most simple tf Autoencoder model
'''
# encoderNodes = [256, 16]
# decoderNodes = [256,256]
# lw = str(encoderNodes).replace(" ", "")+str(decoderNodes).replace(" ", "")
# train_snr = 1.4

Encoder = tf.keras.Sequential([
        # Input Layer
        #layers.Embedding(encoderNodes[0], input_shape=(256,),output_dim=(None,), name='Input'),
        layers.Dense(encoderNodes[0], activation='relu', input_shape=(8,), name='Input'),
        #layers.Dropout(rate=0.01),
        # Hidden Layer
        #layers.BatchNormalization(),
        #layers.Dense(encoderNodes[1], activation='relu', name='EHL1'),
        #layers.BatchNormalization(),
        #layers.Dropout(rate=0.01), 
        # Hidden Layer
        #layers.Dense(encoderNodes[2], activation='relu', name='EHL2'),
        layers.BatchNormalization(),
        # Coded Layer
        layers.Dense(encoderNodes[1], activation='sigmoid', name='Codedfloat'),
        #layers.BatchNormalization(),
        #layers.Lambda(fn.h_norm, input_shape=(n,), output_shape=(n,)),
        #layers.Lambda(fn.normalize, output_shape=(n,)),

], name='1H_Encoder')

NoiseL = tf.keras.Sequential([
        # Noise Layer
        #layers.Lambda(fn.roundCode, input_shape=(n,), output_shape=(n,)),
        #layers.Lambda(fn.BPSK,input_shape=(n,), output_shape=(n,), name='BPSK'),
        #layers.Lambda(fn.roundCode, input_shape=(n,), output_shape=(n,)),
        layers.Lambda(fn.h_norm, input_shape=(n,), output_shape=(n,)),
        layers.Lambda(fn.channel_layer, arguments={'sigma': train_sigma},input_shape=(n,), output_shape=(n,), name='Noise'),
        #layers.GaussianNoise(stddev=np.sqrt(1/(2*train_snr)), 
        #                     input_shape=(n,))
        ], name='Noise')
Decoder = tf.keras.Sequential([ # Array to define layers
        # Adds a densely-connected layer with n units to the model: L1
        #layers.Dense(decoderNodes[0], activation='relu', input_shape=(n,), name='DHL1'),
        # Add another: L2
        #layers.BatchNormalization(),
        #layers.Dense(decoderNodes[1], activation='relu', name='DHL2'),
        # Add another: L3
        #layers.BatchNormalization(),
        #layers.Dense(decoderNodes[2], activation='relu', name='DHL3'),
        # Add layer with k output units:
        #layers.BatchNormalization(),
        layers.Dense(decoderNodes[0], activation='softmax', input_shape=(n,),name='Output')
], name='1H_Decoder')
AAutoencoder = tf.keras.Sequential([Encoder, NoiseL, Decoder])
plot_model(AAutoencoder,to_file='GraphNN/'+title+'/'+title+'_'+lw+'_'+timestr+'.pdf',show_shapes=True)
    
'''
    Overall Settings/ Compilation
'''
#lossFunc = 'binary_crossentropy'
lossFunc = 'categorical_crossentropy'
AAutoencoder.compile(loss=lossFunc,
              optimizer='adam')
'''
    Summaries and checkpoints 
'''
summary = AAutoencoder.summary()
callbacks_list = []
''' 
    Training
'''
start = time.process_time()
history = AAutoencoder.fit(x_train_data, u_train_labels, epochs=numEpochs, 
                   batch_size=batchSize, shuffle=True, verbose=0, callbacks=callbacks_list)
#history = MLNN.fit(x_train, u_train_labels, epochs=numEpochs, batch_size=batchSize,
#          validation_data=(x_val, u_val_labels))
end = time.process_time()

print('The NN has trained ' + str(end - start) + ' s')
# summarize history for loss
trainingFig = plt.figure(figsize=(8, 6), dpi=80)
plt.title('Batch size = '+str(batchSize))
plt.plot(history.history['loss']) # all outputs: ['acc', 'loss', 'val_acc', 'val_loss']
#plt.plot(history.history['acc'])
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
AAutoencoder.save('Models/'+title+'/'+title+'_'+lw+'_Mep_'+str(numEpochs)+'_'+str(train_snr)+'.h5')  # creates a HDF5 file


'''
    Prediction
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
plt.ylim([2*10**-6, 8*10**-1])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('./Results/1HAutoencoder_vs_MAP_'+lw+'_Mep_'+str(numEpochs)+'_'+str(train_snr)+'.png', bbox_inches='tight', dpi=300)
