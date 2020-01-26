# -*- coding: utf-8 -*-
'''
    Prediction
'''
def Predict_1HNN_Kp(globalReps, N, NN1H_KpDecoder):
    print('Parameters:\n\
    globalReps: %d\n\
    N: %d\n' % (globalReps, N))
    Decoder=NN1H_KpDecoder
 
    globalErrorMLNN = np.empty([globalReps, len(pOptions)])
    for i_global in range(globalReps):
        for i_p in range(np.size(pOptions)):
            p = pOptions[i_p]
            pos = np.argmin(np.abs((p - train_ps)))
            cat = np.eye(np.size(train_ps))[pos]
            coded_p = np.ones([N, np.size(train_ps)])*cat
            # print(coded_p)
            '''
                Generate channel Input
            '''
            # Source generator 
            d_test = np.random.randint(0,2,size=(N,k)) 
            ind_test_dec = reduce(lambda a,b: 2*a+b, np.transpose(d_test)) 

            # Encoder        
            u_test = np.zeros((N, n),dtype=bool)
            u_test[:,A] = d_test
            
            c_test = np.zeros((N, n),dtype=bool)
            for iii in range(0,N):
                c_test[iii] = d_f.polar_transform_iter(u_test[iii]) 
            x_test = 1.0*c_test
            ''' 
                Channel
            '''
            xflat = np.reshape(x_test, [-1])
            yflat = fn.BAC(xflat,p,q)
            # Encode p in y
            y_test = np.concatenate([yflat.reshape(N,n), coded_p], axis=1) # noisy codewords
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
    avgMLNNError = np.average(globalErrorMLNN,0)
    '''
        Save Data
    '''
    sys.path[-1]
    filename = sys.path[-1]+'/Data/'+ title+'/'+fileN+'.pickle'
    with open(filename,  'wb') as f:
        pickle.dump(avgMLNNError, f)




