# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:17:20 2020

@author: user
"""
def TensorBAC_input_p(x, q):
    codeword = x[0]
    p = x[1]
    onehot = x[2]
    decision = (1-codeword)*p + q*codeword
    noise = K.random_uniform(shape=K.shape(codeword), minval=0.0, maxval=1.0)
    flip = K.less(noise, decision)
    result =  tf.math.add(tf.multiply(K.cast(flip, dtype='float32'), 1.0), codeword)%2
    return K.concatenate([result, onehot], axis=-1)

def architecture(layerWidth, lossFunc='categorical_crossentropy', name = '3I_Dense_NN', train_q = 0.07, bias=True, summary=True):
    # Input
    input_x = Input(shape=(n,), name='Input_x')
    input_p = Input(shape=(1,), name='Input_p')
    input_onehot = Input(shape=(label_size,), name='Input_onehot')
    input_list = [input_x, input_p, input_onehot]
    # Noise layer
    noiseL = Lambda(TensorBAC_input_p, arguments={'q': train_q,},\
                            output_shape=(n+label_size,), name='Noise')(input_list)
    # Hidden layer
    hl1_1 = Dense(layerWidth[0], activation='relu', name='HL1_1', use_bias=bias)(noiseL)
    # Output layer
    output1 = Dense(layerWidth[1], activation='softmax',name='Output1', use_bias=bias)(hl1_1)
    output = output1
    
    model = Model(inputs=input_list, outputs=output, name=name)

    # Draw model
    direc = sys.path[-1]+'/GraphNN'
    plot_model(model, to_file=direc+'/'+title+'/'+fileN+'.png')

    if summary:
        model.summary()
    model.compile(loss=lossFunc, optimizer='adam')
    return model

def train_feedback(model, input_list, train_labels, numEpochs, batchSize, loops, train_size, verbose=0):
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
    plt.title('Batch size = '+str(batchSize))
    plt.plot(history)
    plt.grid(True, which='both')
    plt.xlabel('$M_{ep}$')
    plt.xscale('log')
    plt.legend([lossFunc + ' loss'])
    plt.show()

    filename = sys.path[-1]+'/GraphNN/'+title+'/'+fileN+'_train_'+str(i)+'.png'
    trainingFig.savefig(filename, bbox_inches='tight', dpi=300)
    
    return model

