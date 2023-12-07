# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import he_normal, glorot_uniform
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout
import numpy as np


class NNFullyConnected:

    ###########################################################################################
    #                                          API                                            #
    ###########################################################################################

    ###############
    # Constructor #
    ###############

    def __init__(
            self,
            layer_dims,  # [n_x, n_h1, .., n_hL, n_y]
            learning_rate,
            num_epochs,
            minibatch_size,
            l2_reg,
            activation='leaky_relu',  # 'leaky_relu' or 'tanh'
            output_activation='relu',
            loss_function='mse',
            dropout=0.0,
            seed=0,
    ):

        # safety checks
        if activation not in ['leaky_relu', 'tanh']:
            raise Exception('Incorrect activation function entered.')

        if output_activation != 'relu' or loss_function != 'mse':
            raise Exception('Incorrect output activation and/or loss function entered.')

        # if len(layer_dims) < 3:
        #     raise Exception('Incorrect architecture, at least one hidden layer required.')

        # seed
        self.seed = seed

        # hyper-parameters
        self.l2_reg = l2_reg
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.layer_dims = layer_dims
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.output_activation = output_activation
        self.weight_init_out = glorot_uniform() #seed=self.seed)

        if activation == 'tanh':
            self.activation = 'tanh'
            self.weight_init = glorot_uniform() #seed=self.seed)
        elif activation == 'leaky_relu':
            self.activation = None
            self.weight_init = he_normal() #seed=self.seed)

        # model
        self.model = self.create_fc_model()

        self.model.compile(
            optimizer=Adam(lr=self.learning_rate),
            loss=self.loss_function
        )

    ##############
    # Prediction #
    ##############

    def prediction(self, x):
        y_hat = self.model.predict(x=x, verbose=0)

        return y_hat

    ############
    # Training #
    ############

    def train(self, x, y, validation_data=None, flag_shuffle=True, verbose=0):
        self.model.fit(
            x=x,
            y=y,
            epochs=self.num_epochs,
            batch_size=self.minibatch_size,
            validation_data=validation_data,
            shuffle=flag_shuffle,
            verbose=verbose)  # 0: off, 1: full, 2: brief

        h = self.model.fit(
            x=x,
            y=y,
            epochs=self.num_epochs,
            batch_size=self.minibatch_size,
            validation_data=validation_data,
            shuffle=flag_shuffle,
            verbose=verbose)
        plt.plot(h.history['loss'])
        #plt.show()

    ###########################################################################################
    #                                      Auxiliary                                          #
    ###########################################################################################

    ############
    # FC Model #
    ############

    def create_fc_model(self):

        # Input and output dims
        n_x = self.layer_dims[0]
        n_y = self.layer_dims[-1]

        # Input layer
        x_input = Input(shape=(n_x,), name='input')

        #  First hidden layer
        x = Dense(
            units=self.layer_dims[1],
            activation=self.activation,
            kernel_initializer=self.weight_init,
            kernel_regularizer=l2(self.l2_reg),
        )(x_input)
        if self.activation is None:
            x = LeakyReLU(alpha=0.01)(x)
        x = Dropout(rate=self.dropout, seed=self.seed)(x)

        #  Other hidden layers (if any)
        for l_dim in self.layer_dims[2:-1]:
            x = Dense(
                units=l_dim,
                activation=self.activation,
                kernel_initializer=self.weight_init,
                kernel_regularizer=l2(self.l2_reg),
            )(x)
            if self.activation is None:
                x = LeakyReLU(alpha=0.01)(x)
            x = Dropout(rate=self.dropout, seed=self.seed)(x)

        # Output layer (NOTE: no regularisation / dropout here)
        y_out = Dense(
            units=n_y,
            activation=self.output_activation,
            kernel_initializer=self.weight_init_out,
        )(x)

        # Model
        return Model(inputs=x_input, outputs=y_out)
