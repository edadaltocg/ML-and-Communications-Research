# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:23:38 2020

@author: user
"""
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
            coded_p = np.ones([N, np.size(train_ps)])*cat
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
            x_test = 1.0*c_test # input 1
            p_test = np.repeat(p,len(x_test)).reshape(-1,1) #input 2
            ''' 
                Channel
            '''
            # xflat = np.reshape(x_test, [-1])
            # yflat = fn.BAC(xflat,p,q)
            # y_test = yflat.reshape(N,n)
            # Encode p in y
            #y_test = np.concatenate([yflat.reshape(N,n), coded_p], axis=1) # noisy codewords
            '''
                MLNN 1H Decoder
            '''
            new_in_1 = x_test
            new_in_2 = p_test
            new_in_3 = coded_p
            prediction = fn.get_class(np.argmax(model.predict([new_in_1, new_in_2, new_in_3]),1),train_ps)
            prediction_p = de_class(prediction, train_ps)
            
            '''
                Error Calculation
            '''
            globalErrorMSE[i_global][i_p] = np.average(MSE(prediction_p, p_test))

    '''
        Error treatment
    '''
    avgMSEError = np.average(globalErrorMSE,0)
    '''
        Save Data
    '''
    sys.path[-1]
    filename = sys.path[-1]+'/Data/'+ title+'/'+fileN+'_MSE.pickle'
    with open(filename,  'wb') as f:
        pickle.dump(avgMSEError, f)
    
    return avgMSEError

