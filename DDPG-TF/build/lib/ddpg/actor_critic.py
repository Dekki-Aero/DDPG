import numpy as np
from tensorflow.keras.initializers import RandomUniform as RU
from tensorflow.keras.layers import Dense, Input, concatenate, BatchNormalization
from tensorflow.keras import Model

class _actor_network():
    def __init__(self, state_dim, action_dim,action_bound_range=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound_range = action_bound_range

    def model(self):
        state = Input(shape=self.state_dim, dtype='float32')
        x = Dense(400, activation='relu',kernel_initializer=RU(-1/np.sqrt(self.state_dim),1/np.sqrt(self.state_dim)))(
            state)  #
        x = Dense(300, activation='relu',kernel_initializer=RU(-1/np.sqrt(400),1/np.sqrt(400)))(x)
        out = Dense(self.action_dim, activation='tanh',kernel_initializer=RU(-0.003,0.003))(x)  #
        #out = tf.multiply(out, self.action_bound_range)
        return Model(inputs=state, outputs=out)


class _critic_network():
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def model(self):
        state = Input(shape=self.state_dim, name='state_input', dtype='float32')
        state_i = Dense(400, activation='relu',kernel_initializer=RU(-1/np.sqrt(self.state_dim),1/np.sqrt(self.state_dim)))(state)
        # state_i = Dense(128)(state) #kernel_initializer=RU(-1/np.sqrt(301),1/np.sqrt(301))

        action = Input(shape=(self.action_dim,), name='action_input')
        x = concatenate([state_i, action])
        x = Dense(300, activation='relu',kernel_initializer=RU(-1/np.sqrt(401),1/np.sqrt(401)))(x)
        out = Dense(1, activation='linear')(x)  # ,kernel_initializer=RU(-0.003,0.003) ,,kernel_regularizer=l2(0.001)
        return Model(inputs=[state, action], outputs=out)
