import numpy as np
from keras.models import Sequential
from keras.layers import Dense

class ModelBuilder(object):

    @staticmethod
    def build_simple_ff(num_nodes, input_size, output_size):
        # initialize an empty network
        model = Sequential()
        model.add(Dense(num_nodes, input_dim=input_size, activation='relu'))
        model.add(Dense(num_nodes, activation='relu'))
        model.add(Dense(output_size, activation='linear')
        model.compile(loss='mse', optimizer='adam')
        return model