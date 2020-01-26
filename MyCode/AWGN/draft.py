# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:11:47 2019

@author: user
"""
x = -0.66*K.ones(shape=(256,16))
shape = fn.func_output_shape(x)
#y = fn.tensorAWGN(x)
y = fn.h_norm(x)
# test = np.sqrt(shape)/(K.sqrt(K.sum(x)))
# test = test*x
#soma = tf.reciprocal(K.sqrt(K.sum(x, 1)))*np.sqrt(shape)
#soma = tf.transpose(tf.multiply(tf.transpose(x), soma))
print(K.eval(y))


