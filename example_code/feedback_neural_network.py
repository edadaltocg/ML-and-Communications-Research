import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.utils import plot_model

from libraries import *

title = 'csi_estimator'
fileN = 'csi_estimator'


def TensorBAC_input_p(x, q):
    codeword = x[0]
    p = x[1]
    onehot = x[2]
    decision = (1 - codeword) * p + q * codeword
    noise = K.random_uniform(shape=K.shape(codeword), minval=0.0, maxval=1.0)
    flip = K.less(noise, decision)
    result = tf.math.add(tf.multiply(K.cast(flip, dtype='float32'), 1.0), codeword) % 2
    return K.concatenate([result, onehot], axis=-1)


def architecture(layerWidth, label_size, lossFunc='categorical_crossentropy', name='3I_Dense_NN', train_q=0.07,
                 bias=True,
                 summary=True):
    # Input
    input_x = Input(shape=(n,), name='Input_x')
    input_p = Input(shape=(1,), name='Input_p')
    input_onehot = Input(shape=(label_size,), name='Input_onehot')
    input_list = [input_x, input_p, input_onehot]
    # Noise layer
    noiseL = Lambda(TensorBAC_input_p, arguments={'q': train_q, }, \
                    output_shape=(n + label_size,), name='Noise')(input_list)
    # Hidden layer
    hl1_1 = Dense(layerWidth[0], activation='relu', name='HL1_1', use_bias=bias)(noiseL)
    # Output layer
    output1 = Dense(layerWidth[1], activation='softmax', name='Output1', use_bias=bias)(hl1_1)
    output = output1

    model = Model(inputs=input_list, outputs=output, name=name)

    # Draw model
    direc = 'GraphNN'
    plot_model(model, to_file=direc + '/' + title + '/' + fileN + '.png')

    if summary:
        model.summary()
    model.compile(loss=lossFunc, optimizer='adam')
    return model


def train_feedback(model, input_list, label_size, train_labels, numEpochs, batchSize, loops, train_size, verbose=0):
    input_x, input_p, input_onehot = input_list
    input_pred = []
    for i in range(train_size):
        line = np.random.random((label_size))
        line /= line.sum()
        input_pred.append(line)
    input_pred = np.array(input_pred)

    history = []
    for i in range(loops):
        ''' 
            Training
        '''
        # Batch size sample:
        train_data_input = [input_x, input_p, input_pred]
        result = model.fit(train_data_input, train_labels, epochs=numEpochs, \
                           batch_size=batchSize, shuffle=False, verbose=verbose)
        history.append(result.history['loss'])
        input_pred = model.predict(train_data_input)

    # summarize history for loss
    trainingFig = plt.figure(figsize=(8, 6), dpi=80)
    plt.title('Batch size = ' + str(batchSize))
    plt.plot(history)
    plt.grid(True, which='both')
    plt.xlabel('$M_{ep}$')
    plt.xscale('log')
    plt.legend(['loss'])
    plt.show()

    filename = 'GraphNN/' + title + '/' + fileN + '_train_' + '.png'
    trainingFig.savefig(filename, bbox_inches='tight', dpi=300)

    return model


def predict_est_p_feedback(globalReps, N, model):
    print('Parameters:\n\
    globalReps: %d\n\
    N: %d\n' % (globalReps, N))
    globalErrorMSE = np.empty([globalReps, len(pOptions)])
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
            x_test = 1.0 * c_test  # input 1
            p_test = np.repeat(p, len(x_test)).reshape(-1, 1)  # input 2
            ''' 
                Channel
            '''
            '''
                MLNN 1H Decoder
            '''
            new_in_1 = x_test
            new_in_2 = p_test
            new_in_3 = coded_p
            prediction = fn.get_class(np.argmax(model.predict([new_in_1, new_in_2, new_in_3]), 1), train_ps)
            prediction_p = fn.de(prediction, train_ps)

            '''
                Error Calculation
            '''
            globalErrorMSE[i_global][i_p] = np.average(fn.MSE(prediction_p, p_test))

    '''
        Error treatment
    '''
    avgMSEError = np.average(globalErrorMSE, 0)
    '''
        Save Data
    '''

    filename = 'Data/' + title + '/' + fileN + '_MSE.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(avgMSEError, f)

    return avgMSEError
