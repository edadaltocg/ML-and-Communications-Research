# -*- coding: utf-8 -*-
#%%
'''
    Hyper parameters - 1H
'''
title = 'NN1H'
layerWidth = [128, n]
lw = str(layerWidth).replace(" ", "")
elevado = 16
numEpochs = 2**elevado
batchSize = 256
train_q = 0.07

globalReps = 1000
N = 1000 # number of messages sent
'''
    Train Networks, Predict Models, Plot Results
'''
train_p = 0.01
#NN1HDecoder = train_1HNN(numEpochs, batchSize, train_p, layerWidth)
filename = sys.path[-1]+'/Models/NN1H/'+title+'_'+lw+'_Mep_'+str(numEpochs)+'_'+str(train_p)+'.h5'
NN1H = keras.models.load_model(filename)
NN1HDecoder = NN1H.layers[1]
Predict_1HNN(globalReps, N, NN1HDecoder)
plot_1HNN()

#%%
'''
    Hyper parameters - GA
'''
title = 'NN1H_Kp'
elevado = 14
numEpochs = 2**elevado
batchSize = 32
train_ps = np.array([0.01, 0.1, 0.3])
ps = str(train_ps).replace(" ", "")
train_q = 0.07
layerWidth = [64,0,n]
lw = str(layerWidth).replace(" ", "")
fileN = title+'_'+lw+'_Mep_'+str(numEpochs)+'_ps_'+ps+'_BS_'+str(batchSize)+'_seed_'+str(s)+'_seed1_'+str(s1)
globalReps = 100
N = 100

NN1H_KpDecoder = train_1HNN_Kp(numEpochs, batchSize, train_q, train_ps, layerWidth)
Predict_1HNN_Kp(globalReps, N, NN1H_KpDecoder)
plot_1HNN_Kp()