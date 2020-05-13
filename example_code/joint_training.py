from architectures import *

title = 'Joint_training'
units_prior = [64, 4]
units_decoder = [64, 2 ** k]
'''
    Train Dataset
'''
train_p_vect = train_ps
label_size = len(train_ps)
# input data
globalReps = 1
N = 10
x = possibleCodewords
# Dataset Memory Buffer
p_buffer = dict()
prior_p_buffer = dict()
label_p_hat_buffer = dict()
label_p_hot_buffer = dict()

x_buffer = dict()
y_buffer = dict()
u_label_buffer = dict()

mse_ML = np.zeros([globalReps, len(pOptions)])
mse_ML_it = np.zeros([globalReps, len(pOptions)])

for i_p in range(len(pOptions)):
    p = pOptions[i_p]

    p_buffer[p] = []
    prior_p_buffer[p] = []
    label_p_hat_buffer[p] = []
    label_p_hot_buffer[p] = []
    x_buffer[p] = []
    u_label_buffer[p] = []
    y_buffer[p] = []

    for i_global in range(globalReps):
        # messages and codewords
        d_test, x_test = generate_channel_input(N, k, n, A)
        x_buffer[p].append(x_test)
        u_label_buffer[p].append(h_encode(d_test))
        # BAC channel
        y_test = fn.BAC(x_test.reshape(-1), p, q).reshape(x_test.shape)
        y_buffer[p].append(y_test)
        # Prior 0
        prior_p = np.ones((1, label_size)) * 1 / label_size

        # ML estimator
        for i_test in range(0, N):
            prior_p_buffer[p].append(prior_p)
            y_test_MAP = y_test[i_test, :]
            p_buffer[p].append(p)
            ML_cond = np.zeros((2 ** k, label_size), dtype=float)
            for i_info in range(0, 2 ** k):
                I_0 = (x[i_info, :] * 1 < 10 ** -6)
                I_1 = (x[i_info, :] * 1 > 1 - 10 ** -6)
                diff_count_0 = np.sum(np.mod(y_test_MAP[I_0] + x[i_info, I_0] * 1, 2))
                diff_count_1 = np.sum(np.mod(y_test_MAP[I_1] + x[i_info, I_1] * 1, 2))
                ML_cond[i_info, :] = ((np.clip(train_p_vect / (1 - train_p_vect), 10 ** -12, 1) ** diff_count_0) *
                                      (np.clip(train_q / (1 - train_q), 10 ** -12, 1) ** diff_count_1) * (
                                              (1 - train_p_vect) ** np.sum(1 * I_0)) *
                                      ((1 - train_q) ** np.sum(1 * I_1)))

            # Find the right estimated channel parameter (prior is always uniform)
            ML_marg = np.sum(ML_cond, 0)
            ind_p_map = np.argmax(ML_marg)
            p_hat_ML = train_p_vect[ind_p_map]
            mse_ML[i_global, i_p] += (p_hat_ML - p) ** 2

            # Calculate new prior
            prior_p = prior_p * ML_marg / (np.sum(prior_p * ML_marg))
            label_p_hat_buffer[p].append(prior_p)
            label_p_hot_buffer[p].append(get_class(p_to_choice(p, train_ps), train_ps))

        # Find the right estimated channel parameter (prior iteratively updated)
        ind_p_map_it = np.argmax(prior_p)
        p_hat_ML_it = train_p_vect[ind_p_map_it]
        mse_ML_it[i_global, i_p] = ((p_hat_ML_it - p) ** 2)

avg_mse_ML = np.mean(mse_ML, 0)
# ML average estimate p error
avg_mse_ML_it = np.mean(mse_ML_it, 0)

# Save ML prediction
filename = 'Data/' + title + '/' + 'ML_memory_buffer_estimate_p_generator_MSE.pickle'
pickle_save_variable(avg_mse_ML_it, filename)

# Plot
# NN estimator
NN_est = 'Data/' + 'NN1H_pEST' + '/' + 'NN1H_pEST_[100,4]_Mep_65536_ps_[0.010.10.30.4]_BS_32_MSE' + '.pickle'
# Perfect estimator
perfect = 'Data/' + title + '/' + 'perfect_estimator.pickle'
legend = ['NN estimator', 'GA estimator', 'ML estimator', ]
x = pOptions
ML = 'Data/' + title + '/' + 'ML_memory_buffer_estimate_p_generator_MSE.pickle'
print(ML)
fileN = 'prior_buffer'
dumpfile = 'Results/' + title + '/ML_est_' + str(globalReps * N) + fileN + '.png'
plot_decoding_curves(NN_est, perfect, ML, x=x, dumpfile=dumpfile, legend=legend, X="p", Y="MSE", log_flag=False)

'''
    Format training data
'''
p_train = dict()
x_train = dict()
u_label_train = dict()
prior_train = dict()
prior_label_train = dict()
y_train = dict()
label_p_hot_train = dict()
for i_p in range(len(pOptions)):
    p = pOptions[i_p]

    p_train[p] = np.array(p_buffer[p]).reshape(-1, 1)
    x_train[p] = np.array(x_buffer[p]).reshape(-1, n)
    y_train[p] = np.array(y_buffer[p]).reshape(-1, n)
    u_label_train[p] = np.array(u_label_buffer[p]).reshape(-1, 2 ** k)

    prior_train[p] = np.array(prior_p_buffer[p]).reshape(-1, label_size)
    prior_label_train[p] = np.array(label_p_hat_buffer[p]).reshape(-1, label_size)
    label_p_hot_train[p] = np.array(label_p_hot_buffer[p]).reshape(-1, label_size)

'''
    Training loop
'''
title = 'Custom'
# Compile the model
CHANNEL = 0
ML_LABEL = 1
if CHANNEL:  # Channel in the NN architecture
    model = joint_architecture(x_shape=(possibleCodewords.shape[0],),
                               prior_shape=(label_size,), train_q=0.07, units_prior=units_prior,
                               units_decoder=units_decoder)
else:
    model = joint_architecture_no_channel(x_shape=(possibleCodewords.shape[0],),
                                          prior_shape=(label_size,), train_q=0.07, units_prior=units_prior,
                                          units_decoder=units_decoder)
model.compile(optimizer='adam',
              loss={'Classifier_prior': 'categorical_crossentropy',
                    'Classifier_decoder': 'categorical_crossentropy'},
              loss_weights={'Classifier_prior': 1,
                            'Classifier_decoder': 1})
numEpochs = 2 ** 14
# Train loop
epoch_loss = dict()
epoch_loss['both'] = []
epoch_loss['prior'] = []
epoch_loss['decoder'] = []
for epoch in range(numEpochs):
    # Batches
    batch_loss = dict()
    batch_loss['both'] = []
    batch_loss['prior'] = []
    batch_loss['decoder'] = []
    for i_p in range(len(pOptions)):
        p = pOptions[i_p]

        p_train_batch = p_train[p]
        x_train_batch = x_train[p]
        y_train_batch = y_train[p]
        u_label_batch = u_label_train[p]
        prior_train_batch = prior_train[p]
        if ML_LABEL:
            prior_label_train_batch = prior_label_train[p]
        else:
            prior_label_train_batch = label_p_hot_train[p]

        # Output: loss_prior+loss_decoder, loss_prior, loss_decoder
        if CHANNEL:  # Random channel during training
            both_l, p_l, d_l = model.train_on_batch([x_train_batch, p_train_batch, prior_train_batch],
                                                    [prior_label_train_batch, u_label_batch])
        else:
            both_l, p_l, d_l = model.train_on_batch([y_train_batch, prior_train_batch],
                                                    [prior_label_train_batch, u_label_batch])
        batch_loss['both'].append(both_l)
        batch_loss['prior'].append(p_l)
        batch_loss['decoder'].append(d_l)
    epoch_loss['both'].append(np.array(batch_loss['both']).mean())
    epoch_loss['prior'].append(np.array(batch_loss['prior']).mean())
    epoch_loss['decoder'].append(np.array(batch_loss['decoder']).mean())

# Save model
fileN = 'Joint_model_' + str(units_prior) + str(units_decoder) + '_Mep_' + str(numEpochs) + '_BS_' + str(N) \
        + '_train_ps_' + str(train_ps) + '_channel_' + str(CHANNEL) + '_ml_label_' + str(ML_LABEL)
filename = 'Models/' + title + '/' + fileN + '.h5'
model.save(filename)

'''
    Loss curves
'''
fig, axes = plt.subplots(4, sharex=True, figsize=(12, 12))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Prior Loss", fontsize=14)
axes[0].plot(epoch_loss['prior'])

axes[1].set_ylabel("Decoder Loss", fontsize=14)
axes[1].plot(epoch_loss['decoder'])

axes[2].set_ylabel("Joint Loss", fontsize=14)
axes[2].plot(epoch_loss['both'])

axes[3].set_ylabel("All", fontsize=14)
axes[3].plot(epoch_loss['prior'])
axes[3].plot(epoch_loss['decoder'])
axes[3].plot(epoch_loss['both'])
axes[3].semilogx()
axes[3].set_xlabel("Epoch", fontsize=14)
axes[3].legend(['Prior', 'Decoder', 'Both'])
axes[3].grid()

plt.show()

filename = 'GraphNN/' + title + '/' + fileN + '.png'
fig.savefig(filename, bbox_inches='tight', dpi=300)

'''
    Prediction
'''
globalReps = 1
N = 100

priorError = np.zeros([globalReps, len(pOptions)])
decoderError = np.zeros([globalReps, len(pOptions)])
for i_global in range(globalReps):
    for i_p in range(len(pOptions)):
        p = pOptions[i_p]
        p_test = np.repeat(p, N).reshape(-1, 1)
        d_test, x_test = generate_channel_input(N, k, n, A)
        y_test = fn.BAC(x_test.reshape(-1), p, q).reshape(x_test.shape)

        prior = np.ones((1, label_size)) * 1 / label_size

        u_hat = []
        p_hat = []
        for i in range(N):
            if CHANNEL:
                prior, pred = model.predict([x_test[i].reshape(1, -1),
                                             p_test[i].reshape(1, -1), prior])
            else:
                prior, pred = model.predict([y_test[i].reshape(1, -1), prior])

            u_hat.append(fn.multipleOneshot2messages(pred, messages))
            p_hat.append(train_ps[np.argmax(prior)])

        u_hat = np.array(u_hat).reshape(-1, k)
        # Error Calculation
        priorError[i_global][i_p] = ((p_hat[-1] - p) ** 2)
        decoderError[i_global][i_p] = fn.bitErrorFunction(u_hat, d_test)

# Error treatment
avgdecoderError = np.average(decoderError, 0)
avgpriorError = np.average(priorError, 0)

# Save Data
fileN = 'decoder_Mep_' + str(numEpochs) + '_BS_' + str(N) + \
        '_train_ps_' + str(train_ps) + '_lw_' + str(units_decoder) + 'round'
filename = 'Data/' + title + '/' + fileN + '.pickle'
pickle_save_variable(avgdecoderError, filename)

fileN2 = 'prior_est_Mep_' + str(numEpochs) + '_BS_' + str(N) + \
         '_train_ps_' + str(train_ps) + '_lw_' + str(units_prior) + 'round'
filename = 'Data/' + title + '/' + fileN2 + '.pickle'
pickle_save_variable(avgpriorError, filename)

'''
    Plot Prior
'''
# NN estimator
NN_est = 'Data/' + 'NN1H_pEST' + '/' + 'NN1H_pEST_[100,4]_Mep_65536_ps_[0.010.10.30.4]_BS_32_MSE' + '.pickle'
# Perfect estimator
perfect = 'Data/' + title + '/' + 'perfect_estimator.pickle'

# Learned estimator
ML = 'Data/' + title + '/' + fileN2 + '.pickle'
legend = ['NN', 'Genie aided', 'Prior', ]
x = pOptions
print(ML)
dumpfile = 'Results/' + title + '/ML_est_' + str(globalReps * N) + fileN2 + '_rounding.png'
plot_decoding_curves(NN_est, perfect, ML, x=x, dumpfile=dumpfile, legend=legend, X="p", Y="MSE", log_flag=False)

'''
    Plot Decoder
'''
legend = ['Joint CSI est. & decoding', 'MAP']
x = pOptions
MAP = 'Data/Joint/MAP.pickle'

decoder = 'Data/' + title + '/' + fileN + '.pickle'
print(decoder)

dumpfile = 'Results/' + title + '/decoder_curve_' + str(globalReps * N) + fileN + '_rounding.png'
plot_decoding_curves(decoder, MAP, x=x, dumpfile=dumpfile, legend=legend, X="Eb/No (dB)", Y="BER")
