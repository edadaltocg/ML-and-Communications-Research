# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 13:58:01 2019

@author: user
"""
'''
       Load libraries
'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import erfc  # complementary error function

import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

import sys
sys.path.append('C:\\Users\\user\\Desktop\\GitHub\\PIR\\MyCode')
import my_functions as fn

import time
import os
from ttictoc import TicToc

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
N = 100 # number of messages sent

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
possibleRealCodewords = fn.BPSK(possibleCodewords)
# Channel
CTYPE = "AWGN"
# Error analysis parameters
SNRdbmin = 0
SNRdbmax = 10
SNR_points = 20

SNR_dB_start_Eb = 0
SNR_dB_stop_Eb = 10
SNR_points = 20
SNR_dB_start_Es = SNR_dB_start_Eb + 10*np.log10(k/n)
SNR_dB_stop_Es = SNR_dB_stop_Eb + 10*np.log10(k/n)

sigma_start = np.sqrt(1/(2*10**(SNR_dB_start_Es/10)))
sigma_stop = np.sqrt(1/(2*10**(SNR_dB_stop_Es/10)))
sigmas  = np.linspace(sigma_start, sigma_stop, SNR_points)

SNR = 1/(2*sigmas**2)

Eb_No_dB = 10*np.log10(1/(2*sigmas**2)) - 10*np.log10(k/n) # signal to noise ration (SNR) in (dB)
Eb_No_lin = 10**(Eb_No_dB/10)
BERBPSK = 0.5*erfc(np.sqrt(k/n*Eb_No_dB)) # analytical BER

# train_SNR_Eb = 1
# train_SNR_Es = train_SNR_Eb + 10*np.log10(k/n)
# train_sigma = np.sqrt(1/(2*10**(train_SNR_Es/10)))

C = 1/2*np.log(1+SNR)# channel capacity
sig = np.sqrt(1/SNR) # scaling factor
#BPSK
A = 1 # dmin = 2A

# Simulation
globalReps = 5000

