from keras.layers import Input, Dense, Lambda, Concatenate
from keras.models import Model

from my_functions import *

'''
    Models Architecture
'''
ga_layerWidth = [64, 2 ** k]
label_size = 4


def ga_architecture(layerWidth, y_shape=(possibleCodewords.shape[0],), p_shape=train_ps.shape):
    input_y = Input(shape=y_shape, name='Input_x')
    input_p_hat = Input(shape=p_shape, name='Input_p_hat')
    input_list = [input_y, input_p_hat]
    # Concatenate prior and y
    concat = Concatenate()(input_list)
    # Dense fully connected classifier
    dense_relu = Dense(layerWidth[0], activation='relu', name='Dense_L')(concat)
    dense_softmax = Dense(layerWidth[1], activation='softmax', name='Classifier')(dense_relu)

    ga_model = Model(inputs=[input_y, input_p_hat], outputs=dense_softmax, name='GA Decoder')
    ga_model.summary()
    return ga_model


def csi_architecture(layerWidth, y_shape=(possibleCodewords.shape[0],), p_shape=train_ps.shape):
    input_y = Input(shape=y_shape, name='Input_x')
    input_p_hat = Input(shape=p_shape, name='Input_p_hat')
    input_list = [input_y, input_p_hat]
    # Concatenate prior and y
    concat = Concatenate()(input_list)
    # Dense fully connected classifier
    dense_relu = Dense(layerWidth[0], activation='relu', name='Dense_L')(concat)
    dense_softmax = Dense(layerWidth[1], activation='softmax', name='Classifier')(dense_relu)

    prior_model = Model(inputs=[input_y, input_p_hat], outputs=dense_softmax, name='CSI Est.')
    prior_model.summary()
    return prior_model


def bac_architecture(x_shape=(possibleCodewords.shape[0],), train_q=0.07):
    input_x = Input(shape=x_shape, name='Input_x')
    input_p = Input(shape=(1,), name='Input_p')
    input_list = [input_x, input_p]
    y = Lambda(BAC_input_p, arguments={'q': train_q}, name='Noise')(input_list)
    bac_model = Model(inputs=[input_x, input_p], outputs=y, name='BAC')
    bac_model.summary()
    return bac_model


def joint_architecture(x_shape=(possibleCodewords.shape[0],),
                       prior_shape=(label_size,), train_q=0.07, units_prior=[64, label_size],
                       units_decoder=[64, 2 ** k], title='joint_architecture'):
    # Inputs
    input_x = Input(shape=x_shape, name='Input_x')
    input_p = Input(shape=(1,), name='Input_p')
    input_prior = Input(shape=prior_shape, name='Input_prior')

    # Noisy BAC channel
    y = Lambda(BAC_input_p, arguments={'q': train_q}, name='Noise')([input_x, input_p])

    # Concatenate prior and y
    concat_prior = Concatenate(name="Concat_prior")([y, input_prior])
    # Dense fully connected CSI classifier
    dense_relu_prior = Dense(units_prior[0], activation='relu',
                             name='Dense_L_prior')(concat_prior)
    softmax_prior = Dense(units_prior[1], activation='softmax',
                          name='Classifier_prior')(dense_relu_prior)  # Output 1

    # Concatenate prior prediciton and y
    concat_decoder = Concatenate(name="Concat_decoder")([y, softmax_prior])
    # Dense fully connected Decoder
    dense_relu_decoder = Dense(units_decoder[0], activation='relu',
                               name='Dense_L_decoder')(concat_decoder)
    softmax_decoder = Dense(units_decoder[1], activation='softmax',
                            name='Classifier_decoder')(dense_relu_decoder)  # Output 2

    # Model API
    model = Model(inputs=[input_x, input_p, input_prior],
                  outputs=[softmax_prior, softmax_decoder],
                  name='Joint_CSI_Decoder')
    model.summary()
    # plot model
    plot_model(model, 'GraphNN/' + title + '/Joint_CSI_Decoder_model.png')
    return model


def joint_architecture_no_channel(x_shape=(possibleCodewords.shape[0],),
                                  prior_shape=(label_size,), train_q=0.07, units_prior=[64, label_size],
                                  units_decoder=[64, 2 ** k], title='joint_architecture_no_channel'):
    # Inputs
    input_y = Input(shape=x_shape, name='Input_y')
    input_prior = Input(shape=prior_shape, name='Input_prior')

    # Concatenate prior and y
    concat_prior = Concatenate(name="Concat_prior")([input_y, input_prior])
    # Dense fully connected CSI classifier
    dense_relu_prior = Dense(units_prior[0], activation='relu',
                             name='Dense_L_prior')(concat_prior)
    softmax_prior = Dense(units_prior[1], activation='softmax',
                          name='Classifier_prior')(dense_relu_prior)  # Output 1
    # Round layer
    rounded = Lambda(round_tensor, name='Round')(softmax_prior)
    # Concatenate prior prediciton and y
    concat_decoder = Concatenate(name="Concat_decoder")([input_y, softmax_prior])
    # Dense fully connected Decoder
    dense_relu_decoder = Dense(units_decoder[0], activation='relu',
                               name='Dense_L_decoder')(concat_decoder)
    softmax_decoder = Dense(units_decoder[1], activation='softmax',
                            name='Classifier_decoder')(dense_relu_decoder)  # Output 2

    # Model API
    model = Model(inputs=[input_y, input_prior],
                  outputs=[softmax_prior, softmax_decoder],
                  name='Joint_CSI_Decoder')
    model.summary()
    # plot model
    plot_model(model, sys.path[-1] + '/GraphNN/' + title + '/Joint_CSI_Decoder_no_channel_model.png')
    return model
