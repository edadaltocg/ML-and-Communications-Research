'''
    Prediction
'''
def Predict_1HNN(globalReps, N, NN1HDecoder):
    print('Parameters:\n\
    globalReps: %d\n\
    N: %d\n' % (globalReps, N))
    Decoder=NN1HDecoder
    #globalReps = 1000
    #N = 1000 # number of messages sent
    globalErrorMLNN = np.empty([globalReps, len(pOptions)])
    print(len(pOptions))
    for i_global in range(globalReps):
        for i_p in range(np.size(pOptions)):
            p = pOptions[i_p]
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
            y_test = yflat.reshape(N,n) # noisy codewords
            '''
                MLNN 1H Decoder
            '''
            prediction = np.round(Decoder.predict(y_test))
            u_hat = fn.multipleOneshot2messages(prediction, messages)
            
            '''
                Error Calculation
            '''
            # globalError[i_global][i_p] = fn.codeErrorFunction(y_test, x_test)
            globalErrorMLNN[i_global][i_p] = fn.bitErrorFunction(u_hat, d_test)

    '''
         Error treatment
    '''
   
    avgMLNNError = np.average(globalErrorMLNN,0)
    '''
        Save Data
    '''
    sys.path[-1]
    filename = sys.path[-1]+'/Data/'+ title+'/'+ title+'_'+lw+'_Mep_'+str(numEpochs)+\
        '_'+str(train_p)+'.pickle'
    with open(filename,  'wb') as f:
        pickle.dump(avgMLNNError, f)