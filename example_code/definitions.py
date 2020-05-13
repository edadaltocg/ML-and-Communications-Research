"""
Created on Mon Dec  9 09:42:45 2019

@author: meryem benammar
"""
import numpy as np


def half_adder(a, b):
    s = a ^ b
    c = a & b
    return s, c


def full_adder(a, b, c):
    s = (a ^ b) ^ c
    c = (a & b) | (c & (a ^ b))
    return s, c


def add_bool(a, b):
    if len(a) != len(b):
        raise ValueError('arrays with different length')
    k = len(a)
    s = np.zeros(k, dtype=bool)
    c = False
    for i in reversed(range(0, k)):
        s[i], c = full_adder(a[i], b[i], c)
    if c:
        warnings.warn("Addition overflow!")
    return s


def inc_bool(a):
    k = len(a)
    increment = np.hstack((np.zeros(k - 1, dtype=bool), np.ones(1, dtype=bool)))
    a = add_bool(a, increment)
    return a


def bitrevorder(x):
    m = np.amax(x)
    n = np.ceil(np.log2(m)).astype(int)
    for i in range(0, len(x)):
        x[i] = int('{:0{n}b}'.format(x[i], n=n)[::-1], 2)
    return x


def int2bin(x, N):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        binary = np.zeros((len(x), N), dtype='bool')
        for i in range(0, len(x)):
            binary[i] = np.array([int(j) for j in bin(x[i])[2:].zfill(N)])
    else:
        binary = np.array([int(j) for j in bin(x)[2:].zfill(N)], dtype=bool)

    return binary


def bin2int(b):
    if isinstance(b[0], list):
        integer = np.zeros((len(b),), dtype=int)
        for i in range(0, len(b)):
            out = 0
            for bit in b[i]:
                out = (out << 1) | bit
            integer[i] = out
    elif isinstance(b, np.ndarray):
        if len(b.shape) == 1:
            out = 0
            for bit in b:
                out = (out << 1) | bit
            integer = out
        else:
            integer = np.zeros((b.shape[0],), dtype=int)
            for i in range(0, b.shape[0]):
                out = 0
                for bit in b[i]:
                    out = (out << 1) | bit
                integer[i] = out

    return integer


# Returns the frozen bit vector
def polar_design_awgn(N, k, design_snr_dB):
    S = 10 ** (float(design_snr_dB) / float(10))
    z0 = np.zeros(N)

    z0[0] = np.exp(-S)
    for j in range(1, int(np.log2(N)) + 1):
        u = 2 ** j
        for t in range(0, int(u / 2)):
            T = z0[t]
            z0[t] = 2 * T - T ** 2  # upper channel
            z0[int(u / 2) + t] = T ** 2  # lower channel

    # sort into increasing order
    idx = np.argsort(z0)

    # select k best channels
    idx = np.sort(bitrevorder(idx[0:k]))

    A = np.zeros(N, dtype=bool)
    A[idx] = True

    return A


def polar_design_awgn_arikan(N, k, design_EbN0_dB):
    S = 10 ** (float(design_EbN0_dB) / float(10))
    z0 = np.zeros(N)

    z0[0] = np.exp(-(float(k) / float(N)) * S)
    for j in range(1, int(np.log2(N)) + 1):
        u = 2 ** j
        for t in range(0, int(u / 2)):
            T = z0[t]
            z0[t] = 2 * T - T ** 2  # upper channel
            z0[int(u / 2) + t] = T ** 2  # lower channel

    # sort into increasing order
    idx = np.argsort(z0)

    # select k worst channels
    idx = np.sort(idx[k:N])

    A = np.ones(N, dtype=bool)
    A[idx] = False

    return A


# Encodes a binary stream (with frozen bits) to polar coded  symbols
def polar_transform_iter(u):
    N = len(u)
    n = 1
    x = np.copy(u)
    stages = np.log2(N).astype(int)
    for s in range(0, stages):
        i = 0
        while i < N:
            for j in range(0, n):
                idx = i + j
                x[idx] = x[idx] ^ x[idx + n]
            i = i + 2 * n
        n = 2 * n
    return x
