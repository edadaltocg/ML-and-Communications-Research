# -*- coding: utf-8 -*-
#%%
'''
    Simulation vs theory
'''
markerlist = ['x', '', '', '^']
linelist = ['', '-', '--', '']
colorlist = ['k', 'k', 'k', 'k'] 
fig = fn.plotAWGN([Eb_No_dB,Eb_No_dB, Eb_No_dB, Eb_No_dB], [ 
              codederrorBPSK, theoreticalErrorBPSK, avgMAPError, avgHardError], 
            ['0.5 rate BPSK', 'Uncoded BPSK', 'Soft MAP Decoder', 'Hard MAP Decoder'],
            colorlist, linelist, markerlist,
            lineWidth, markerSize)
plt.xlim([SNRdbmin, SNRdbmax])
plt.ylim([10**-5, 1])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('Results/simu-vs-theory.png', bbox_inches='tight', dpi=300)


#%% Plot and save results
'''
    Ploting MAP
'''
filename = sys.path[-1]+'/Data/MAP/MAP.pickle'
with open(filename, 'rb') as f:
    avgMAPError = pickle.load(f)
filename = sys.path[-1]+'/Data/MAP/Global_error.pickle'
with open(filename, 'rb') as f:
    avgGlobalError = pickle.load(f)
markerlist = ['', '', '^']
linelist = ['-', '--', '']
colorlist = ['k', 'k', 'k']
fig = fn.plotAWGN([pOptions, pOptions],
            [avgMAPError, avgGlobalError], 
            ['MAP Decoder', 'No decoding'],
            colorlist, linelist, markerlist,
            lineWidth, markerSize, X='p',Y='BER')
            
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
filename = sys.path[-1]+'/Results/MAP'+str(globalReps*N)+'.png'
fig.savefig(filename, bbox_inches='tight', dpi=300)
#%%
'''
    Single DNN 1H decoder predict vs MAP
'''
def plot_1HNN():
    filename = sys.path[-1]+'/Data/'+ title+'/'+ title+'_'+lw+'_Mep_'+str(numEpochs)+\
        '_'+str(train_p)+'.pickle'
    print('Plotting: ', filename)
    with open(filename, 'rb') as f:
        avgMLNNError = pickle.load(f)
    filename = sys.path[-1]+'/Data/MAP/MAP.pickle'
    with open(filename, 'rb') as f:
        avgMAPError = pickle.load(f)
    filename = sys.path[-1]+'/Data/MAP/Global_error.pickle'
    with open(filename, 'rb') as f:
        avgGlobalError = pickle.load(f)
    markerlist = ['', '', '^']
    linelist = ['-', '--', '']
    colorlist = ['k', 'k', 'k']
    fig = fn.plotAWGN([pOptions, pOptions],
                [avgMAPError, avgMLNNError], 
                ['MAP Decoder', '1H NN, p = '+str(train_p)],
                colorlist, linelist, markerlist,
                lineWidth, markerSize, X='p, q = '+str(train_q), Y='BER')
                
    plt.show()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    fig.set_size_inches(width, height)
    filename = sys.path[-1]+'/Results/MLNN_vs_Decoder_'+lw+'_Mep_'\
        +str(numEpochs)+'_'+str(train_p)+'_'+str(train_q)+'.png'
    fig.savefig(filename, bbox_inches='tight', dpi=300)

#%%%
'''
    Multiple BER Plot NN1H
'''
title = 'NN1H'
lw = '[128,16]'
numEpochs = 2**16
train_p_opt = [0.01, 0.1, 0.4]
avgMLNNError = []

for train_p in train_p_opt:
    filename = sys.path[-1]+'/Data/'+ title+'/'+ title+'_'+lw+'_Mep_'+str(numEpochs)+'_'+str(train_p)+'.pickle'
    with open(filename, 'rb') as f:
        avgMLNNError.append(pickle.load(f))
# pOptionsMAP = np.array([0.005, 0.025, 0.05,0.075, 0.1, 0.2, 0.3, 0.4, 0.5])
# pOptionsNN1H = np.array([0, 0.005, 0.025, 0.05,0.075, 0.1, 0.2, 0.3, 0.4, 0.5])
avgMLNNError1 = avgMLNNError[0]
avgMLNNError2 = avgMLNNError[1]
avgMLNNError3 = avgMLNNError[2]
markerlist = ['', 'x','o','^','+']
linelist = ['--', '-', '-', '-', ''] 
colorlist = ['k', 'k', 'k', 'k', 'k']
fig = fn.plotAWGN([pOptions,pOptions,pOptions,pOptions],
            [avgMAPError, avgMLNNError1, avgMLNNError2, avgMLNNError3], 
            ['Soft MAP', '$p_{train} = 0.01$', '$p_{train} = 0.1$', '$p_{train} = 0.4$'],
            colorlist, linelist, markerlist,
            lineWidth, markerSize, X='p, q = 0.07')
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig(sys.path[-1]+'/Results/MAP_vs_'+title+'_'+lw+'for_all_train_p.png', bbox_inches='tight', dpi=300)

#%%
'''
    Ploting - GA
'''
def plot_1HNN_Kp():
    filename = sys.path[-1]+'/Data/'+ title+'/'+fileN+'.pickle'
    print('Plotting: ', filename)
    with open(filename, 'rb') as f:
        avgMLNNError = pickle.load(f)
    filename = sys.path[-1]+'/Data/MAP/MAP.pickle'
    with open(filename, 'rb') as f:
        avgMAPError = pickle.load(f)
    filename = sys.path[-1]+'/Data/MAP/Global_error.pickle'
    with open(filename, 'rb') as f:
        avgGlobalError = pickle.load(f)
    markerlist = ['', '', '^']
    linelist = ['-', '--', '']
    colorlist = ['k', 'k', 'k']
    fig = fn.plotAWGN([pOptions, pOptions],
                [avgMAPError, avgMLNNError], 
                ['MAP Decoder', '1H NN Kp, train_p = '+ps],
                colorlist, linelist, markerlist,
                lineWidth, markerSize, X='p, q = '+str(train_q), Y='BER')
                
    plt.show()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    fig.set_size_inches(width, height)
    filename = sys.path[-1]+'/Results/'+title+'/'+title+'_vs_Decoder_'+fileN+'.png'
    fig.savefig(filename, bbox_inches='tight', dpi=300)

