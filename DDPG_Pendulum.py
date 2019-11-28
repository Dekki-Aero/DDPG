import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, concatenate, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD, Adam
import pandas as pd
from collections import deque
import random
import numpy as np
import gym
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm

from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import RandomUniform as RU

# same as test gym 2
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
        out = tf.multiply(out, self.action_bound_range)
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


class DDPG():
    def __init__(self, env, actor=None, critic=None, buffer=None,action_bound_range=2):
        #############################################
        # --------------- Parametres-----------------#
        #############################################
        self.max_buffer_size = 100000
        self.batch_size = 64
        self.T = 300  ## Time limit for a episode
        self.tow = 0.001  ## Soft Target Update
        self.gamma = 0.99  ## discount factor
        # self.target_update_freq = 10  ## frequency for updating target weights
        self.explore_time = 1000
        self.act_learning_rate = 0.0001
        self.critic_learning_rate = 0.001
        self.dflt_dtype = 'float32'
        self.n_episodes = 400

        self.actor_opt = Adam(self.act_learning_rate)
        self.critic_opt = Adam(self.critic_learning_rate)
        self.r, self.l, self.qlss = [], [], []
        self.env = env
        action_dim = 1
        state_dim = len(env.reset())
        if buffer is not None:
            print('using loaded models')
            self.buffer = buffer
            self.actor = actor
            self.critic = critic
        else:
            self.buffer = deque(maxlen=self.max_buffer_size)
            self.actor = _actor_network(state_dim, action_dim,action_bound_range).model()
            self.critic = _critic_network(state_dim, action_dim).model()
        self.actor_target = _actor_network(state_dim, action_dim,action_bound_range).model()
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target = _critic_network(state_dim, action_dim).model()
        self.critic.compile(loss='mse', optimizer=self.critic_opt)
        self.critic_target.set_weights(self.critic.get_weights())

    #############################################
    # --------------Replay Buffer----------------#
    #############################################

    def add_experience(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample_batch(self):
        replay_buffer = np.array(random.sample(self.buffer, self.batch_size))
        arr = np.array(replay_buffer)
        states_batch = np.vstack(arr[:, 0])
        actions_batch = arr[:, 1].astype(self.dflt_dtype).reshape(-1, 1)
        rewards_batch = arr[:, 2].astype(self.dflt_dtype).reshape(-1, 1)
        next_states_batch = np.vstack(arr[:, 3])
        done_batch = np.vstack(arr[:, 4]).astype(bool)
        return states_batch, actions_batch, rewards_batch, next_states_batch, done_batch

    #############################################
    # ----Action based on exploration policy-----#
    #############################################
    def take_action(self, state, rand):
        actn = self.actor.predict(state).ravel()[0]
        if rand:
            return actn + random.uniform(-2, 2)
        else:
            return actn

    #############################################
    # --------------Update Networks--------------#
    #############################################
    def train_networks(self, states_batch, actions_batch, rewards_batch, next_states_batch, done_batch):
        next_actions = self.actor_target(next_states_batch)

        q_t_pls_1 = self.critic_target([next_states_batch, next_actions])
        y_i = rewards_batch
        for i in range(self.batch_size):
            if not done_batch[i]:
                y_i[i] += q_t_pls_1[i] * self.gamma

        self.critic.train_on_batch([states_batch, actions_batch], y_i)

        with tf.GradientTape() as tape:
            a = self.actor(states_batch)
            tape.watch(a)
            q = self.critic([states_batch, a])
        dq_da = tape.gradient(q, a)

        with tf.GradientTape() as tape:
            a = self.actor(states_batch)
            theta = self.actor.trainable_variables
        da_dtheta = tape.gradient(a, theta, output_gradients=-dq_da)
        self.actor_opt.apply_gradients(zip(da_dtheta, self.actor.trainable_variables))

    def update_target(self, target, online, tow):
        init_weights = online.get_weights()
        update_weights = target.get_weights()
        weights = []
        for i in tf.range(len(init_weights)):
            weights.append(tow * init_weights[i] + (1 - tow) * update_weights[i])
        target.set_weights(weights)
        return target

    def train(self, n_episodes):
        obs = self.env.reset()
        state_dim = len(obs)
        experience_cnt = 0
        self.ac = []
        rand = True
        for episode in range(n_episodes):
            ri, li, qlssi = [], [], []
            state_t = np.array(self.env.reset(), dtype=self.dflt_dtype).reshape(1, state_dim)
            state_t[0][2] /= 8
            for t in range(self.T):
                action_t = self.take_action(state_t, rand)
                # action_t = action_t
                self.ac.append(action_t)
                temp = self.env.step([action_t])  # step returns obs_t+1, reward, done
                state_t_pls_1, rwrd_t, done_t = temp[0], temp[1], temp[2]
                state_t_pls_1[2] /= 8
                ri.append(rwrd_t)
                self.buffer.append(
                    [state_t.ravel(), action_t, rwrd_t, np.array(state_t_pls_1, self.dflt_dtype), done_t])

                state_t = np.array(state_t_pls_1, dtype=self.dflt_dtype).reshape(1, state_dim)
                if not rand:
                    states_batch, actions_batch, rewards_batch, next_states_batch, done_batch = self.sample_batch()
                    self.train_networks(states_batch, actions_batch, rewards_batch, next_states_batch, done_batch)
                    self.actor_target = self.update_target(self.actor_target, self.actor, self.tow)
                    self.critic_target = self.update_target(self.critic_target, self.critic, self.tow)
                if done_t:
                    rr = np.sum(ri)
                    self.r.append(rr)
                    print('Episode %d : Total Reward = %f' % (episode, rr))
                    self.qlss.append(qlssi)
                    plt.plot(self.r)
                    plt.pause(0.0001)
                    break
                if rand: experience_cnt += 1
                if experience_cnt > self.explore_time: rand = False

            '''if episode % 500 ==0 :
                self.actor.save('actor_model.h5')
                self.critic.save('critic_model.h5')
                self.actor_target.save('actor_model.h5')
                self.critic_target.save('critic_model.h5')
                with open('buffer','wb') as file :
                    pickle.dump({'buffer':self.buffer},file)'''


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    '''actor = load_model('actor_model.h5')
    critic = load_model('critic_model.h5')
    with open('buffer','rb') as file:
        buffer = pickle.load(file)['buffer']'''

    ddpg = DDPG(env)  # actor=actor,critic=critic,buffer=buffer
    ddpg.train(ddpg.n_episodes)