# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 17:50:31 2020

@author: user
"""

u_train_labels = fn.messages2onehot(1*messages.copy())
x_train_data = possibleCodewords.copy()
u_train_labels = np.repeat(u_train_labels, 10, axis=0)
x_train_data = np.repeat(x_train_data, 10, axis=0)
trainSize = np.size(x_train_data, 0)

x = x_train_data
p = 0.01
q = 0.07
#noise = K.random_uniform(shape=(fn.func_output_shape(x),), minval=0.0, maxval=1.0)
noise = np.random.uniform(size=np.shape(x), low=0, high=1)
#decision = tf.map_fn(lambda x: (1-x)*p + q*x, x)
decision = (1-x)*p + q*x
#flip = K.less(noise, decision)   
flip = (noise<decision)*1
#K.cast(flip, dtype='int32')
#result =  tf.math.add(K.cast(flip, dtype='float32'), x)%2
result = (flip + x)%2

filename = './Models/NN1H/NN1H_[64,16]_Mep_16384_01.h5'
NN1H = keras.models.load_model(filename)

