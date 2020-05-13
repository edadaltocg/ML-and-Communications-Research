import time

import tensorflow as tf
from tensorflow.keras import layers
from libraries import *

title = 'GA'
fileN = 'genie_aided_decoder'

'''
    DNN One Hot Model Decoder with Channel Knowledge
'''


def train_1HNN_Kp(numEpochs, batchSize, train_q, train_ps, layerWidth):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    lw = str(layerWidth)
    print('Parameters:\n\
    numEpochs: %d\n\
    batchSize: %d\n\
    train_q: %f\n\
    train_ps: %s\n\
    lw: %s' % (numEpochs, batchSize, train_q, train_ps, lw))
    # Training parameters
    u_train_labels = fn.messages2onehot(messages.copy())
    x_train_data = possibleCodewords.copy() * 1.0

    u_train_labels = np.repeat(u_train_labels, 20, axis=0)
    x_train_data = np.repeat(x_train_data, 20, axis=0)
    trainSize = np.size(x_train_data, 0)
    '''
        Architecture
    '''
    NoiseL = tf.keras.Sequential([
        # Noise Layer
        layers.Lambda(fn.TensorBACconcat, arguments={'q': train_q, 'train_ps': train_ps}, \
                      input_shape=(n,), output_shape=(n + np.size(train_ps),), name='Noise')
    ], name='Noise')
    NN1H_KpDecoder = tf.keras.Sequential([  # Array to define layers
        layers.Dense(layerWidth[0], input_shape=(n + np.size(train_ps),), activation='relu', name='1H_HL_1'),
        layers.Dense(layerWidth[1], input_shape=(n + np.size(train_ps),), activation='softmax', name='1H_Output')
    ], name='1H_Decoder')
    NN1H = tf.keras.Sequential([NoiseL, NN1H_KpDecoder])

    '''
        Overall Settings/ Compilation
    '''
    lossFunc = 'mse'
    NN1H.compile(loss=lossFunc, optimizer='adam')
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
    plt.title('Batch size = ' + str(batchSize))
    plt.plot(history.history['loss'])  # all outputs: ['acc', 'loss', 'val_acc', 'val_loss']
    plt.grid(True, which='both')
    plt.xlabel('$M_{ep}$')
    plt.xscale('log')
    plt.legend([lossFunc + ' loss'])
    plt.show()

    filename = 'GraphNN/' + title + '/' + fileN + '_train' + timestr + '.png'
    trainingFig.savefig(filename, bbox_inches='tight', dpi=300)

    '''
        Saving model
    '''
    filename = 'Models/' + title + '/' + fileN + '.h5'
    NN1H.save(filename)  # creates a HDF5 file

    return NN1H_KpDecoder


'''
    Prediction
'''


def Predict_1HNN_Kp(globalReps, N, NN1H_KpDecoder):
    print('Parameters:\n\
    globalReps: %d\n\
    N: %d\n' % (globalReps, N))
    Decoder = NN1H_KpDecoder

    globalErrorMLNN = np.empty([globalReps, len(pOptions)])
    for i_global in range(globalReps):
        for i_p in range(np.size(pOptions)):
            p = pOptions[i_p]
            pos = np.argmin(np.abs((p - train_ps)))
            cat = np.eye(np.size(train_ps))[pos]
            coded_p = np.ones([N, np.size(train_ps)]) * cat
            '''
                Generate channel Input
            '''
            # Source generator
            d_test = np.random.randint(0, 2, size=(N, k))
            ind_test_dec = np.reduce(lambda a, b: 2 * a + b, np.transpose(d_test))

            # Encoder
            u_test = np.zeros((N, n), dtype=bool)
            u_test[:, A] = d_test

            c_test = np.zeros((N, n), dtype=bool)
            for iii in range(0, N):
                c_test[iii] = d_f.polar_transform_iter(u_test[iii])
            x_test = 1.0 * c_test
            ''' 
                Channel
            '''
            xflat = np.reshape(x_test, [-1])
            yflat = fn.BAC(xflat, p, q)
            # Encode p in y
            y_test = np.concatenate([yflat.reshape(N, n), coded_p], axis=1)  # noisy codewords
            '''
                MLNN 1H Decoder
            '''
            prediction = np.round(Decoder.predict(y_test))
            u_hat = fn.multipleOneshot2messages(prediction, messages)

            '''
                Error Calculation
            '''
            globalErrorMLNN[i_global][i_p] = fn.bitErrorFunction(u_hat, d_test)

    '''
        Error treatment
    '''
    avgMLNNError = np.average(globalErrorMLNN, 0)
    '''
        Save Data
    '''
    filename = 'Data/' + title + '/' + fileN + '.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(avgMLNNError, f)
