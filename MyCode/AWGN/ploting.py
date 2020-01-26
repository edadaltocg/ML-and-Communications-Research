# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 20:20:19 2019

@author: user
"""
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
    MAP Decoder vs theory
'''
markerlist = ['^', '']
linelist = ['', '--']
colorlist = ['k', 'k']
fig = fn.plotAWGN([Eb_No_dB,Eb_No_dB], [avgMAPError, 
             theoreticalErrorBPSK], 
            ['Soft MAP Decoder', 'Uncoded BPSK'],
            colorlist, linelist, markerlist,
            lineWidth, markerSize)
plt.xlim([SNRdbmin, SNRdbmax])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('Results/MAP-vs-theory.png', bbox_inches='tight', dpi=300)

#%%
'''
    Single decoder predict vs MAP
'''
markerlist = ['', '^']
linelist = ['--', '-']
colorlist = ['k', 'k']
fig = fn.plotAWGN([Eb_No_dB,Eb_No_dB], [avgMAPError, 
             avgMLNNError], 
            ['Soft MAP Decoder', '$M_{ep} = 2^{12}$'],
            colorlist, linelist, markerlist,
            lineWidth, markerSize)
plt.xlim([SNRdbmin, SNRdbmax])
plt.ylim([10**-5, 10**-1])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('Results/MLNN-vs-Decoder-Mep-'+str(numEpochs)+'.png', bbox_inches='tight', dpi=300)

#%%%
'''
    Multiple BER Plot MLNN
'''
#lm = '[128,64,32,256]'
title = 'MLNN'
lm = '[1024,512,256,8]'
mep = [2**12, 2**14, 2**16, 2**18]
filename = './Data/'+title+'/'+title+'_'+lm+'_Mep_'+str(mep[0])+'.pickle'
with open(filename, 'rb') as f:
    avgMLNNError1 = pickle.load(f)
filename = './Data/'+title+'/'+title+'_'+lm+'_Mep_'+str(mep[1])+'.pickle'
with open(filename, 'rb') as f:
    avgMLNNError2 = pickle.load(f)
filename = './Data/'+title+'/'+title+'_'+lm+'_Mep_'+str(mep[2])+'.pickle'
with open(filename, 'rb') as f:
    avgMLNNError3 = pickle.load(f)
filename = './Data/'+title+'/'+title+'_'+lm+'_Mep_'+str(mep[3])+'.pickle'
with open(filename, 'rb') as f:
     avgMLNNError4 = pickle.load(f)
    
markerlist = ['','x','o','+','^']
linelist = ['--', '-','-', '-', '-', '-'] 
colorlist = ['k', 'k','k', 'k', 'k', 'k']
fig = fn.plotAWGN([Eb_No_dB,Eb_No_dB,Eb_No_dB,Eb_No_dB, Eb_No_dB],
            [avgMAPError, avgMLNNError1, avgMLNNError2, avgMLNNError3, avgMLNNError4], 
            ['Soft MAP', '$M_{ep} = 2^{12}$', '$M_{ep} = 2^{14}$', '$M_{ep} = 2^{16}$','$M_{ep} = 2^{18}$'],
            colorlist, linelist, markerlist,
            lineWidth, markerSize)
plt.xlim([SNRdbmin, SNRdbmax])
plt.ylim([10**-5, 8*10**-1])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('Results/MAP_vs_'+title+'_'+lm+'.png', bbox_inches='tight', dpi=300)

#%%
'''
    Multiple BER Plot NN1H
'''
#lm = '[128,64,32,256]'
title = 'NN1H'
lm = '[128,64,32,256]'
mep = [2**12, 2**14, 2**16, 2**18]
filename = './Data/'+title+'/'+title+'_'+lm+'_Mep_'+str(mep[0])+'_1.pickle'
with open(filename, 'rb') as f:
    avgMLNNError1 = pickle.load(f)
filename = './Data/'+title+'/'+title+'_'+lm+'_Mep_'+str(mep[1])+'_1.pickle'
with open(filename, 'rb') as f:
    avgMLNNError2 = pickle.load(f)
filename = './Data/'+title+'/'+title+'_'+lm+'_Mep_'+str(mep[2])+'_1.pickle'
with open(filename, 'rb') as f:
    avgMLNNError3 = pickle.load(f)
filename = './Data/'+title+'/'+title+'_'+lm+'_Mep_'+str(mep[3])+'_1.pickle'
with open(filename, 'rb') as f:
      avgMLNNError4 = pickle.load(f)
    
markerlist = ['', 'x','o','+','^']
linelist = ['--', '-', '-', '-', '-'] 
colorlist = ['k', 'k', 'k', 'k', 'k']
fig = fn.plotAWGN([Eb_No_dB,Eb_No_dB,Eb_No_dB,Eb_No_dB, Eb_No_dB],
            [avgMAPError, avgMLNNError1, avgMLNNError2, avgMLNNError3, avgMLNNError4], 
            ['Soft MAP', '$M_{ep} = 2^{12}$', '$M_{ep} = 2^{14}$', '$M_{ep} = 2^{16}$', '$M_{ep} = 2^{18}$'],
            colorlist, linelist, markerlist,
            lineWidth, markerSize)
plt.xlim([SNRdbmin, SNRdbmax])
plt.ylim([10**-5, 8*10**-1])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('Results/MAP_vs_'+title+'_'+lm+'.png', bbox_inches='tight', dpi=300)

#%%
'''
    Encoder vs MAP
'''
filename = './Data/AEncoder/AEncoder_[128,64,16]_Mep_65536_1.pickle'
with open(filename, 'rb') as f:
    avgEncoderError = pickle.load(f)
markerlist = ['','', 'o']
linelist = ['-','--', '-']
colorlist = ['k', 'k', 'k']
fig = fn.scatterAWGN([Eb_No_dB,Eb_No_dB,Eb_No_dB], 
                     [avgGlobalError, avgMAPError, 
             avgEncoderError], 
            ['Uncoded BPSK (8,8)', 'Soft MAP Decoder', 'DNN Encoder + MAP'],
            colorlist, linelist, markerlist,
            lineWidth, markerSize)
plt.xlim([SNRdbmin, SNRdbmax])
plt.ylim([2*10**-6, 10**-1])
plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
fig.set_size_inches(width, height)
fig.savefig('Results/AEncoder_vs_MAP_[128,64,16]_Mep_65536_1.png', bbox_inches='tight', dpi=300)
