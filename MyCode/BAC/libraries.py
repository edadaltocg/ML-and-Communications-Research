# -*- coding: utf-8 -*-
'''
       Load libraries
'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import erfc  # complementary error function
from functools import reduce

import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import Adam

import sys
sys.path.append('/content/drive/My Drive/BAC/BAC')
import my_functions as fn

import time
import os
#from ttictoc import TicToc
#!pip install ttictoc
import pickle

#Plot setup
#plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams["font.family"] = "Times New Roman"
letterSize = 8
markerSize = 3
lineWidth = 0.75

mpl.rc('xtick', labelsize=letterSize)
mpl.rc('ytick', labelsize=letterSize)
mpl.rc('axes', labelsize=letterSize)
mpl.rc('legend', fontsize=letterSize)

# width as measured in inkscape
width = 3.487
height = width / 1.618


#import matplotlib
#del matplotlib.font_manager.weight_dict['roman']
#matplotlib.font_manager._rebuild()

'''
       Load constants
'''
N = 1000 # number of messages sent

# Message characteristics
## Hamming coding
m = 4 # parity bits
n = 2**m-1 # total bits of one codeword
k = n-m # data bits
n = 16
k = 8
d = n-k+1
R = k/n # rate
name = np.array([n, k]) # Hamming(n,k)

Gdef = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
       [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
       [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
G = Gdef

messages, possibleCodewords = fn.possibleCodewordsG(name, G)

# Channel
CTYPE = "BAC"
# BAC
pOptions = np.array([0.005, 0.025, 0.05,0.075, 0.1, 0.2, 0.3, 0.4, 0.5])

q = 0.07
# Simulation
globalReps = 100
N = 1000 # number of messages sent

# Redefining code matrix
import Definitions as d_f
k = 4               # Number of input bits 
n = 16              # Number of coded bits 

design_EbN0_dB = 0 
design_snr_dB = design_EbN0_dB + 10*np.log10(float(k)/float(N)) 

Arikan_polar = 1    # Choose between Arikan's polar code, or Gruber's polar

######################################################################## 
#           Frozen bits and codebook construction
######################################################################## 

# Create all possible information words (array encoding)
d = np.zeros((2**k,k),dtype=bool)
for i in range(1,2**k):
    d[i]= d_f.inc_bool(d[i-1])
messages = d
# Choose either Gruber's polar code or Arikan's construction of frozen bits A 
if Arikan_polar == 0: 
    A = d_f.polar_design_awgn(n, k, design_snr_dB)
    
else : # (Arikan's frozen bits obtained from Matlab's polar code construction)
    A = d_f.polar_design_awgn_arikan(n, k, design_EbN0_dB)  
    
# Creating all possible codewords and storing them for the MAP decoder
u = np.zeros((2**k,n),dtype=bool)
u[:,A] = d

x = np.zeros((2**k,n),dtype=bool)
for i in range(0,2**k):
    x[i] = d_f.polar_transform_iter(u[i])
possibleCodewords = x