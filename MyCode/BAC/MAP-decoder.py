globalReps = 100
N = 1000 # number of messages sent
globalError = np.empty([globalReps, len(pOptions)])
globalErrorMAP = np.empty([globalReps, len(pOptions)])
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
        x_test = 1*c_test
        ''' 
            Channel
        '''
        xflat = np.reshape(x_test, [-1])
        yflat = fn.BAC(xflat,p,q)
        y_test = yflat.reshape(N,n) # noisy codewords
        y = y_test
        '''
            MAP Decoder 
        '''
        MAP = np.empty([N,k])
        for i in range(N): # for each received message y
            f_MAP = np.empty([2**k])
            for j in range(2**k): # For each possible message
                I0, I1 = fn.getI(possibleCodewords[j]*1) # matrices for all codewords
                x0, x1 = fn.getx0x1(possibleCodewords[j]*1, I0, I1)
                y0, y1 = fn.getx0x1(y[i], I0, I1)    
                f_MAP[j] = (np.log(1-p)*fn.card(I0) + np.log(1-q)*fn.card(I1) + \
                        np.log(p/(1-p))*np.sum(fn.XOR(x0,y0)) + np.log(q/(1-q))*np.sum(fn.XOR(x1, y1)))
            MAP[i] = messages[np.argmax(f_MAP)]
        '''
            Error Calculation
        '''
        globalError[i_global][i_p] = fn.codeErrorFunction(y, x_test)
        globalErrorMAP[i_global][i_p] = fn.bitErrorFunction(MAP, d_test)

'''
    Error treatment
'''
avgMAPError = np.average(globalErrorMAP, 0)
avgGlobalError = np.average(globalError, 0)
'''
    Save Data
'''
filename = sys.path[-1]+'/Data/MAP/MAP.pickle'
with open(filename,  'wb') as f:
    pickle.dump(avgMAPError, f)

filename = sys.path[-1]+'/Data/MAP/Global_error.pickle'
with open(filename,  'wb') as f:
    pickle.dump(avgGlobalError, f)
    