'''
    DNN One Hot Model Decoder
'''
'''
    Training and validation data
'''
def train_1HNN(numEpochs, batchSize, train_p, layerWidth):
    
    print('Parameters:\n\
    numEpochs: %d\n\
    batchSize: %d\n\
    train_p: %f\n\
    lw: %s' % (numEpochs, batchSize, train_p, lw) )
    #Training parameters
    #title = 'NN1H'
    timestr = time.strftime("%Y%m%d-%H%M%S")
    #elevado = 14
    #numEpochs = 2**elevado
    #batchSize = 256

    u_train_labels = fn.messages2onehot(messages.copy())
    x_train_data = possibleCodewords.copy()

    u_train_labels = np.repeat(u_train_labels, 100, axis=0)
    x_train_data = np.repeat(x_train_data, 100, axis=0)
    trainSize = np.size(x_train_data, 0)

    #train_p = 0.01
    train_q = 0.07
    '''
        Architecture
    '''
    #layerWidth = [64, n]
    
    NoiseL = tf.keras.Sequential([
            # Noise Layer
            layers.Lambda(TensorBAC, arguments={'p': train_p, 'q': train_q},input_shape=(n,), output_shape=(n,), name='Noise')
            ], name='Noise')
    NN1HDecoder = tf.keras.Sequential([ # Array to define layers
            layers.Dense(layerWidth[0], activation='relu', input_shape=(n,),name='1H_HL'),
            layers.Dense(layerWidth[1], activation='softmax', input_shape=(n,),name='1H_Output')
    ], name='1H_Decoder')
    NN1H = tf.keras.Sequential([NoiseL, NN1HDecoder])
    #plot_model(NN1H,to_file='GraphNN/'+title+'/'+title+'_'+lw+'_'+timestr+'.pdf',show_shapes=True)

    '''
        Overall Settings/ Compilation
    '''
    lossFunc = 'binary_crossentropy'
    NN1H.compile(loss=lossFunc ,\
                optimizer='Adam')
    '''
        Summaries and checkpoints 
    '''
    #summary = NN1H.summary()
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

    filename = sys.path[-1]+'/GraphNN/'+title+'/'+title+'_train'+ timestr+'.png'
    trainingFig.savefig(filename, bbox_inches='tight', dpi=300)

    '''
        Saving model
    '''
    filename = sys.path[-1]+'/Models/'+title+'/'+title+'_'+lw+'_Mep_'+str(numEpochs)+\
            '_'+str(train_p)+'.h5'
    NN1H.save(filename)  # creates a HDF5 file

    return NN1HDecoder
