# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import he_normal, glorot_uniform
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, LeakyReLU, MaxPool1D, Flatten


class My1DConv:

    ###########################################################################################
    #                                          API                                            #
    ###########################################################################################

    ###############
    # Constructor #
    ###############

    def __init__(
            self,
            cnn_input_dims,  # tuple (n_timesteps, n_features)
            kernel_size,
            pool_size,
            num_filters,
            learning_rate,
            num_epochs,
            minibatch_size,
            fc_units,
            output,
            fc_l2_reg,
            conv_stride=1,
            pool_stride=2,
            fc_dropout=0.0,
            fc_activation='leaky_relu',
            output_activation='relu',
            loss_function='mse',
            seed=0
    ):

        # safety checks
        if fc_activation not in ['leaky_relu', 'tanh']:
            raise Exception('Incorrect activation function entered.')

        if output_activation == 'sigmoid':
            raise Exception('Incorrect output activation function entered.')

        if output_activation == 'softmax' and loss_function != 'categorical_crossentropy':
            raise Exception('Incorrect loss function entered.')

        # seed
        self.seed = seed

        # hyper-parameters
        self.output = output
        self.fc_l2_reg = fc_l2_reg
        self.fc_dropout = fc_dropout
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.num_filters = num_filters
        self.conv_stride = conv_stride
        self.pool_stride = pool_stride,
        self.fc_units = fc_units
        self.num_epochs = num_epochs
        self.cnn_input_dims = cnn_input_dims
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.output_activation = output_activation
        self.weight_init_out = glorot_uniform()

        if fc_activation == 'tanh':
            self.fc_activation = 'tanh'
            self.fc_weight_init = glorot_uniform()
        elif fc_activation == 'leaky_relu':
            self.fc_activation = None
            self.fc_weight_init = he_normal()

        # model
        self.model = self.create_1dcnn_model()

        self.model.compile(
            optimizer=Adam(lr=self.learning_rate),
            loss=self.loss_function
        )

    ##############
    # Prediction #
    ##############

    def prediction(self, x):
        y_hat = self.model.predict(x=x, verbose=0)  # probability of each class

        return y_hat

    ############
    # Training #
    ############

    def train(self, x, y, validation_data=None, flag_shuffle=False, verbose=0):
        self.model.fit(
            x=x,
            y=y,
            epochs=self.num_epochs,
            batch_size=self.minibatch_size,
            validation_data=validation_data,
            shuffle=flag_shuffle,
            verbose=verbose  # 0: off, 1: full, 2: brief
        )

    ###########################################################################################
    #                                      Auxiliary                                          #
    ###########################################################################################

    ############
    # FC Model #
    ############

    def create_1dcnn_model(self):
        # input shape
        n_timesteps, n_features = self.cnn_input_dims

        # Input layer
        inputs = Input(shape=(n_timesteps, n_features))

        # 1DConv layer
        # TODO: add regularization / dropout
        x = Conv1D(
            filters=self.num_filters,
            kernel_size=self.kernel_size,
            strides=self.conv_stride
        )(inputs)

        # Max Pooling layer for 1D
        x = MaxPool1D(
            pool_size=self.pool_size,
            strides=self.pool_stride
        )(x)

        x = Flatten()(x)

        # FC layer
        x = Dense(
            units=self.fc_units,
            activation=self.fc_activation,
            kernel_initializer=self.fc_weight_init,
            kernel_regularizer=l2(self.fc_l2_reg),
        )(x)
        if self.fc_activation is None:
            x = LeakyReLU(alpha=0.01)(x)
        x = Dropout(rate=self.fc_dropout, seed=self.seed)(x)

        # Output layer (NOTE: no regularisation / dropout here)
        y_out = Dense(
            units=self.output,
            activation=self.output_activation,
            kernel_initializer=self.weight_init_out,
        )(x)

        # Model
        return Model(inputs=inputs, outputs=y_out)
