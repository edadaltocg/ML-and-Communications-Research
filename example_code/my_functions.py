import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.utils import plot_model, to_categorical
from numpy import *
from scipy import stats
from scipy.special import erfc  # complementary error function
from libraries import *


def plotit(t, u, title="Title", X="time (s)", Y="Amplitude"):
    plt.figure()
    plt.plot(t, u, 'b')

    plt.grid(True)

    plt.title(title)
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.show()
    return


def scatterit(t, u, title="Title", X="time (s)", Y="Amplitude"):
    plt.figure()
    plt.plot(t, u, 'o')

    plt.grid(True)

    plt.title(title)
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.show()
    return


def sineWave(t, mean, amplitude, f):
    omega = 2 * pi * f  # fs > 2f
    return amplitude * sin(omega * t + 0) + mean


def analog2binary(t, bCodification, inputSignal):
    # Example: we divide the amplitude in a scale of 8 bits and each sequence of 8 bits represents a point
    # in the end we should have t/T * 8 bits
    u = inputSignal
    M = len(u)
    x = 2 ** bCodification * (
            u - mean(u) - min(u - mean(u))) / max(u - mean(u) - min(u - mean(u)))

    x = x.astype(int)

    scatterit(t, x, "Quantized signal")
    b = ''  # strings of all bits togheter
    for i in range(M):
        str = "{0:b}".format(x[i])  # variable length source code word
        str = str.zfill(bCodification)
        b = b + str  # transform b in an array of symbols 0's and 1's

    return b


def PSK2(b, V):
    # 2 - PSK: 2 different phase representations. Maps binary symbols unsing the map 0 -> -1 and 1-> 1. Constellation of size 2
    um = empty(len(b))  # u - modulated (symbols modulated)
    for i in range(len(b)):  # step function
        if (b[i] == '0'):
            um[i] = -1 * V
        else:
            um[i] = 1 * V

    return um


def pulseSignal(A, d):
    # d is the discretization of the pulse
    pulse1 = A * ones(int(d / 2))
    pulse2 = zeros(int(d / 2))

    return concatenate((pulse1, pulse2), axis=None)


def constantSignal(A, d):
    # d is the discretization of the pulse
    pulse1 = A * ones(int(d))
    return pulse1


def pulses2waveform(N, d, p, uk):
    uwv = empty(N * d)
    for i in range(len(uwv)):
        uwv[i] = uk[int(i / d)] * p[i % d]
    return uwv


def fourriertransform(x, interval, fc, BB):
    freq = fft.fftfreq(len(x), interval)  # Frequency space
    xhat = fft.fft(x)
    fig, ax = plt.subplots()

    ax.plot(freq, xhat.real)
    ax.set_title('Waveform Fourrier Tranform')
    ax.set_xlabel('Frequency in Hertz [Hz]')
    ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
    ax.set_xlim(-fc - BB, fc + BB)

    return concatenate((freq, xhat), axis=0)


def Q(x):
    return 0.5 * erfc(x / sqrt(2))


def MPSK_BER(M, Eb, No):
    n = log2(8)
    Es = n * Eb  # energy per symbol: nEb where 2^n = M
    gamma_s = Es / No
    return 2 * Q(sqrt(2 * gamma_s) * sin(pi / M)) / n


def empiricBER(u, uhat):
    n = len(u)
    indicatrice = u != uhat
    P = sum(indicatrice) / n
    return P


def binarySignalPower(u):
    n = len(u)
    return sum(u ** 2) / n


def BSC(b, p):
    decision = random.rand(len(b))
    noise = decision < p
    return (noise + b) % 2


def bitEnergy(Eb, N0):
    return Eb / N0


def Hb(p):
    return -p * log2(p) - (1 - p) * log2(1 - p)


def matrixGenerator(H, name):
    n = name[0]
    k = name[1]
    Ik = eye(k)
    P = H[:, :n - (n - k)].T
    G = concatenate((Ik, P), axis=1)
    return G


def parityMatrix(name):  # arbitrary rule
    n = name[0]
    k = name[1]
    n_p = n - k  # number of parity bits
    limit = k - n_p + 1
    e = eye(k)  # message generator
    P = empty([k, n - k])
    for l in range(0, k):
        b = e[l]
        p = empty(n_p)
        for j in range(0, n_p):
            aux = b[j:limit + j]
            p[j] = sum(aux) % 2
        P[l] = p
    return P


def parityCheckMatrix(name):
    n = name[0]
    k = name[1]
    m = n - k
    tuples = asarray(list(itertools.product(*[(0, 1)] * m)))
    H = tuples[1:].T
    soma = sum(H, 0)
    counter = size(H, 1)
    i = 0
    while i < counter:
        if (soma[i] <= 1):
            H = delete(H, i, 1)
            counter -= 1
            i -= 1
            soma = sum(H, 0)
        i += 1
    return concatenate((H, eye(m)), axis=1)


def parityCheck(G):
    k = size(G, 0)
    n = size(G, 1)
    I = eye(n - k)
    P = G[:k, k:]

    H = concatenate((-P.T % 2, I), axis=1)
    C = dot(G, H.T)
    return print('Parity check: \n', sum(C))


def hammingDistance(a, b):
    return np.sum(a != b, axis=1)


def possibleCodewordsH(name):
    n = name[0]
    k = name[1]
    H = parityCheckMatrix(name)
    tuples = asarray(list(itertools.product(*[(0, 1)] * n)))

    counter = size(tuples, 0)
    i = 0
    while i < counter:
        if (sum(dot(tuples[i], H.T) % 2) != 0):
            tuples = delete(tuples, i, 0)
            counter -= 1
            i -= 1
        i += 1
    return tuples


def possibleCodewordsG(name, G):
    n = name[0]
    k = name[1]
    tuples = asarray(list(itertools.product(*[(0, 1)] * k)))
    words = empty([size(tuples, 0), n])
    for i in range(size(tuples, 0)):
        words[i] = dot(tuples[i], G) % 2
    return tuples, words


def syndrome(y, H, name):
    n = name[0]
    k = name[1]
    N = size(y, 0)
    S = empty([N, n - k])  # identify if there are errors - syndrome
    # k stored syndromes
    for i in range(N):
        S[i] = dot(y[i], H.T) % 2  # syndrome row vector
    return S


def codeErrorFunction(y, x):
    Ecw = sum(y != x)  # Codeword Error with hamming decoding
    return Ecw / size(x)


def bitErrorFunction(uhat, u):
    Eb = sum(uhat != u)
    return Eb / size(u)


def generateU(N, k):
    return stats.bernoulli.rvs(0.5, size=[N, k])  # input message matrix


def generteCodeWord(N, n, u, G):
    x = empty([N, n])  # code words
    for i in range(N):
        x[i] = dot(u[i], G) % 2  # codeword row vector
    return x


###################
# AWGN
###################
def BPSK(x):
    return x * 2 - 1


def decodeAWGN(x):
    return (np.sign(x) == 1).astype(float)


def hardMAPAWGN(x, messages, possibleCodewords):
    y = (np.sign(x) == 1).astype(float)
    minDistWord = np.argmin(hammingDistance(possibleCodewords, y), 0)  # find word of minimum distance
    MAP = messages[minDistWord]
    return MAP


def h_norm_np(x):
    return (x - np.mean(x)) / np.sqrt(np.var(x))


def linear2dB(x):
    return 10 * np.log(x)


def AWGN(x, snr):
    sigma = np.sqrt(1 / (2 * snr))  # scaling factor
    n = np.size(x)
    noise = np.random.normal(0, sigma, n)
    return x + noise


def euclidianDistance(possibleCodewords, y):
    return np.linalg.norm(possibleCodewords - y, ord=2, axis=1)


def plotAWGN(xlist, ylist, legend, colorlist, linelist, markerlist, lineWidth, markerSize, X="Eb/No (dB)", Y="BER"):
    fig = plt.figure(figsize=(8, 6))

    for i, array in enumerate(xlist):
        plt.plot(xlist[i], ylist[i], color=colorlist[i], linewidth=lineWidth,
                 linestyle=linelist[i], marker=markerlist[i], markersize=markerSize)

    plt.grid(True, which="both")
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.yscale('log')
    plt.legend(legend)
    return fig


def scatterAWGN(xlist, ylist, legend, colorlist, linelist, markerlist, lineWidth, markerSize, X="Eb/No (dB)", Y="BER"):
    fig = plt.figure(figsize=(8, 6))
    for j in range(2):
        plt.plot(xlist[j], ylist[j], color=colorlist[j], linewidth=lineWidth,
                 linestyle=linelist[j], marker=markerlist[j], markersize=markerSize)
    for i in range(2, len(xlist)):
        plt.scatter(xlist[i], ylist[i], color=colorlist[i],
                    marker=markerlist[i], s=markerSize, zorder=2)

    plt.grid(True, which="both")
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.yscale('log')
    plt.legend(legend)
    return fig


###################
# DNNs
###################

def tensorAWGN(x):
    train_snr = 1  # Article: Gruber - polar codes
    # snr = K.constant(train_snr,dtype=tf.float32)
    noise = K.random_normal(shape=K.shape(x), mean=0., stddev=np.sqrt(1 / (2 * train_snr)))
    # noise = K.random_normal_variable(shape=(func_output_shape(x),), mean=0, scale=K.sqrt(1/(2*snr)))
    # noiseFloat = K.cast(noise, dtype=tf.float32)
    result = tf.math.add(noise, x)
    return result


def channel_layer(x, sigma):
    w = K.random_normal(K.shape(x), mean=0.0, stddev=sigma)
    return x + w


def normalize(x):  # |x_i| <= 1
    xmin = K.min(x)
    xmax = K.max(x)
    a = K.constant(-1)
    b = K.constant(1)
    return a + ((x - xmin) * (b - a)) / (xmax - xmin)


def h_norm(x):
    return (x - K.mean(x)) / K.sqrt(K.var(x))


def func_output_shape(x):
    shape = x.get_shape().as_list()[1]
    return shape


def metricBER(y_true, y_pred):
    return K.mean(K.not_equal(y_true, y_pred))


def ber(y_true, y_pred):
    return K.mean(K.cast(K.not_equal(y_true, K.round(y_pred)), dtype='float32'))


def metricBER1H(y_true, y_pred):
    # return K.mean(K.not_equal(K.argmax(y_true),K.argmax(y_pred)))
    return K.mean(K.not_equal(y_true, K.round(y_pred)))  # ????


def tensorPossibleMessages(name):
    n = name[0]
    k = name[1]
    nptuples = asarray(list(itertools.product(*[(0, 1)] * k)))
    ttuples = K.variable(nptuples)
    return ttuples


###################
# One hot message encoding
###################


def messages2onehot(u):
    n = u.shape[0]
    k = u.shape[1]
    N = 2 ** k
    index = np.zeros(N)
    encoded = np.zeros([n, N])
    for j in range(n):
        for i in range(k - 1, -1, -1):
            index[j] = index[j] + u[j][i] * 2 ** (k - 1 - i)
        encoded[j][int(index[j])] = 1
    return encoded


def singleMessage2onehot(m):
    k = m.shape[0]
    n = 2 ** k
    encoded = np.zeros(n)
    index = 0
    for i in range(k - 1, -1, -1):
        index = index + m[i] * 2 ** (k - 1 - i)
    encoded[int(index)] = 1
    return encoded


def onehot2singleMessage(h, messages):
    index = np.argmax(h)
    return messages[index]


def multipleOneshot2messages(h, messages):
    indexes = np.argmax(h, 1)
    n = h.shape[1]
    k = int(np.log2(n))
    N = len(indexes)
    out = np.zeros([N, k])
    for i in range(N):
        out[i] = messages[indexes[i]]
    return out


def TensorOnehot2singleMessage(h):
    # todo
    index = tf.argmax(h)
    return np.asarray([int(x) for x in list('{0:08b}'.format(index))])


def roundCode(x):
    return tf.stop_gradient(K.round(x) - x) + x


def signCode(x):
    return K.sign(x)


def createDir(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


def messages2customEncoding(messages, Encoder):
    return Encoder.predict(messages)


def clipp(x):
    return K.clip(x, -1, 1)


################################
# BAC functions
################################
def BAC(b, p, q):
    noise = random.rand(len(b))
    decision = np.array([p if i == 0 else q for i in b])
    flip = noise < decision
    return (1.0 * flip + b) % 2


def BAC_Kp(b, p, q):
    noise = random.rand(len(b))
    decision = np.array([p if i == 0 else q for i in b])
    flip = noise < decision
    return (flip + b) % 2


def TensorBAC(x, p, q):
    noise = K.random_uniform(shape=(fn.func_output_shape(x),), minval=0.0, maxval=1.0)
    decision = tf.map_fn(lambda x: (1 - x) * p + q * x, x)
    flip = K.less(noise, decision)
    K.cast(flip, dtype='int32')
    result = tf.math.add(K.cast(flip, dtype='float32'), x) % 2
    return result


def generate_channel_input(N, k, n, A):
    '''
    Generates a random channel input.
    - Input parameters:
        N - number of messages
        k - number of bits per message
        n - size of the codewords
        A - encoding matrix
    - Output:
        d_test - generated messages
        x_test - codewords of size n
    '''
    # Source generator
    d_test = np.random.randint(0, 2, size=(N, k))
    ind_test_dec = np.reduce(lambda a, b: 2 * a + b, np.transpose(d_test))

    # Encoder
    u_test = np.zeros((N, n), dtype=bool)
    u_test[:, A] = d_test

    c_test = np.zeros((N, n), dtype=bool)
    for iii in range(0, N):
        c_test[iii] = d_f.polar_transform_iter(u_test[iii])
    x_test = 1.0 * c_test

    return d_test, x_test


def BAC_channel_wrapper(p, q, x_test):
    '''
    Take an input of shape (N,n) and return a noisy output of a BAC(p,q) of the
    same shape.
    - Input parameters:
        p - BAC channel statistics p
        q - BAC channel statistics q
        x - codewords
    - Output:
        y - noisy codewords
    '''
    xflat = np.reshape(x_test, [-1])
    yflat = BAC(xflat, p, q)
    y_test = yflat.reshape(x_test.shape)  # noisy codewords
    return y_test


def getI(x):  # Make sets I0 and I1
    I0 = x == 0
    I1 = x == 1
    return I0, I1


def card(I):  # Cardinality of a set
    return np.sum(I)


def XOR(x, y):  # XOR operation
    return (x + y) % 2


def getx0x1(x, I0, I1):  # make x0 and x1 based on I0 and I1, resp.
    return x[I0], x[I1]


def BAC_MAP_decoder(N, k, possibleCodewords, messages, y, p, q):
    '''
    MAP decoder of the BAC channel.
    - Input parameters:
        N - total number of received messages
        k - number of bits per message
        possibleCodewords - possible codewords
        messages - possible messages
        y - noisy codewords
    - Output:
        MAP - estimated messages
    '''
    MAP = np.empty([N, k])
    for i in range(N):  # for each received message y
        f_MAP = np.empty([2 ** k])
        for j in range(2 ** k):  # For each possible message
            I0, I1 = getI(possibleCodewords[j] * 1)  # matrices for all codewords
            x0, x1 = getx0x1(possibleCodewords[j] * 1, I0, I1)
            y0, y1 = getx0x1(y[i], I0, I1)
            f_MAP[j] = (np.log(1 - p) * card(I0) + np.log(1 - q) * card(I1) + \
                        np.log(p / (1 - p)) * np.sum(XOR(x0, y0)) + np.log(q / (1 - q)) * np.sum(XOR(x1, y1)))
        MAP[i] = messages[np.argmax(f_MAP)]
    return MAP


def pickle_save_variable(x, filename):
    '''
    Save a variable in a pickle file in filename.
    '''
    with open(filename, 'wb') as f:
        pickle.dump(x, f)


def pickle_load_variable(filename):
    '''
    Load a pickle variable from filename.
    '''
    with open(filename, 'rb') as f:
        x = pickle.load(f)
    return x


def plot_decoding_curves(*filenames, x, dumpfile, legend, X="Eb/No (dB)", Y="BER", log_flag=True):
    '''
    Log BER performance of different decoders.
    - Input parameters:
        *filenames - variables to load and plot
        x - x axis of reference
    - Output:
        Figure display and saved figure in png format in dumpfile
    '''
    fig = plt.figure(figsize=(width, height))
    for i, filename in enumerate(filenames):
        array = pickle_load_variable(filename)
        plt.plot(x, array, color=colorlist[i], linewidth=lineWidth,
                 linestyle=linelist[i], marker=markerlist[i], markersize=markerSize)
    plt.legend(legend)

    plt.grid(True, which="both")
    plt.xlabel(X)
    plt.ylabel(Y)
    if (log_flag):
        plt.yscale('log')
    fig.set_size_inches(width, height)
    plt.show()
    fig.savefig(dumpfile, bbox_inches='tight', dpi=300)


def plot_training_curve(history, batchSize, numEpochs, title):
    '''
    Plot and save training loss curve
    '''
    fig = plt.figure(figsize=(width, height), dpi=80)
    plt.title('Batch size = ' + str(batchSize))
    plt.plot(history)
    plt.grid(True, which='both')
    plt.xlabel('$M_{ep}$')
    plt.xscale('log')
    plt.show()
    fileN = title + str(np.random.randint(10)) + '_numEpochs_' + str(numEpochs)
    filename = 'GraphNN/' + title + '/' + fileN + '.png'
    fig.savefig(filename, bbox_inches='tight', dpi=300)


def BAC_input_p(x, q):
    '''
    Tensor function to simulate a BAC(p, q) channel.
    - Input parameters:
        x - list of codewords and p of the channel
        q - q of the channel
    - Output:
        y - noisy codeword in a tensor
    '''
    codeword = x[0]
    p = x[1]
    decision = (1 - codeword) * p + q * codeword
    noise = K.random_uniform(shape=K.shape(codeword), minval=0.0, maxval=1.0)
    flip = K.less(noise, decision)
    result = tf.math.add(tf.multiply(K.cast(flip, dtype='float32'), 1.0), codeword) % 2
    return result


def p_to_choice(p, train_ps):
    '''
    Convert p to the least distant category
    '''
    return np.argmin(np.abs(train_ps - p))


def get_class(choice, opts):
    '''
        Get an one hot class from an array of classes
    '''
    classes = np.array(range(np.size(opts)))
    M = to_categorical(classes, num_classes=np.size(classes), dtype='float32')
    return M[choice]


def get_multiple_class(choice, opts):
    '''
    Get multiple one hot categories from an array of options.
    '''
    output_ = []
    for i in choice:
        output_.append(get_class(i, opts))
    return np.array(output_)


def h_encode(u, messages):
    classes = []
    for _ in u:
        classes.append(np.argmin(np.sum(np.abs(_ - messages), 1)))
    return [np.identity(len(messages))[clas] for clas in classes]


def h_decode(uh, messages):
    return [messages[np.argmax(_)] for _ in uh]


def round_tensor(x):
    return x + tf.stop_gradient(tf.math.floordiv(x, tf.math.reduce_max(x)) - x)


###############################
# MSE CURVES
###############################
def genie_aided(est_p, train_ps):
    return np.min(np.abs(est_p - train_ps), 1)


def reshape_column(x):
    return x.reshape(-1, 1)


def SSE(x, y):  # Sufficient Statistic Estimator
    return 1 + q - 2 * (1 - np.mean(y))


def NN_est():
    return 0


def MSE(A, B):
    return ((A - B) ** 2).mean(axis=1)
