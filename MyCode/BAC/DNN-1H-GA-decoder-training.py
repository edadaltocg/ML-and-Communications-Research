# -*- coding: utf-8 -*-
'''
    DNN One Hot Model Decoder with Channel Knowledge
'''
def train_1HNN_Kp(numEpochs, batchSize, train_q, train_ps, layerWidth):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print('Parameters:\n\
    numEpochs: %d\n\
    batchSize: %d\n\
    train_q: %f\n\
    train_ps: %s\n\
    lw: %s' % (numEpochs, batchSize, train_q, train_ps, lw) )
    #Training parameters
    u_train_labels = fn.messages2onehot(messages.copy())
    x_train_data = possibleCodewords.copy()*1.0

    u_train_labels = np.repeat(u_train_labels, 20, axis=0)
    x_train_data = np.repeat(x_train_data, 20, axis=0)
    trainSize = np.size(x_train_data, 0)
    '''
        Architecture
    '''
    NoiseL = tf.keras.Sequential([
            # Noise Layer
            layers.Lambda(TensorBACconcat, arguments={'q': train_q, 'train_ps': train_ps},\
                        input_shape=(n,), output_shape=(n+np.size(train_ps),), name='Noise')
            ], name='Noise')
    NN1H_KpDecoder = tf.keras.Sequential([ # Array to define layers
            layers.Dense(layerWidth[0], input_shape=(n+np.size(train_ps),), activation='relu',name='1H_HL_1'),
            # layers.Dense(layerWidth[1], input_shape=(n+np.size(train_ps),), activation='relu',name='1H_HL_2'),
            layers.Dense(layerWidth[2], input_shape=(n+np.size(train_ps),), activation='softmax',name='1H_Output')
    ], name='1H_Decoder')
    NN1H = tf.keras.Sequential([NoiseL, NN1H_KpDecoder])
    #plot_model(NN1H,to_file='GraphNN/'+title+'/'+title+'_'+lw+'_'+timestr+'.pdf',show_shapes=True)

    '''
        Overall Settings/ Compilation
    '''
    lossFunc = 'mse'
    NN1H.compile(loss=lossFunc ,\
                optimizer='adam')
    '''
        Summaries and checkpoints 
    '''
    summary = NN1H.summary()
    callbacks_list = []
    ''' 
        Training
    '''
    history = NN1H.fit(x_train_data, u_train_labels, epochs=numEpochs, \
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

    filename = sys.path[-1]+'/GraphNN/'+title+'/'+fileN+'_train'+ timestr+'.png'
    trainingFig.savefig(filename, bbox_inches='tight', dpi=300)

    '''
        Saving model
    '''
    filename = sys.path[-1]+'/Models/'+title+'/'+fileN+'.h5'
    NN1H.save(filename)  # creates a HDF5 file

    return NN1H_KpDecoder


