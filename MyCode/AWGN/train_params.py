# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:07:27 2019

@author: user
"""

#%% 1H Training
batchSize = 256

u_train_labels = fn.messages2onehot(messages.copy())
x_train_data = possibleCodewords.copy()

u_train_labels = np.repeat(u_train_labels, 10, axis=0)
x_train_data = np.repeat(x_train_data, 10, axis=0)
trainSize = np.size(x_train_data, 0)

train_snr = 1
train_sigma=np.sqrt(1/(2*train_snr))

#%% Array Training
batchSize = 256

u_train_labels = messages.copy()
x_train_data = possibleCodewords.copy()

u_train_labels = np.repeat(u_train_labels, 1, axis=0)
x_train_data = np.repeat(x_train_data, 1, axis=0)
trainSize = np.size(x_train_data, 0)

train_snr = 1
train_sigma=np.sqrt(1/(2*train_snr))
