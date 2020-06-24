import tensorflow as tf
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os
import random

#Class for the Agent
class Agent(object):
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        #Respective Ratios for all the machines
        #self.m1 = m1
        #self.m2 = m2
        #self.m3 = m3

        self.memory = deque(maxlen=10000)

        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.learning_rate = 0.001

        self.model = self._build_model()


    def _build_model(self):
        model = Sequential()

        #Adding the dense layers
        #Might take time if not run on a GPU
        model.add(Dense(64,input_shape=(self.state_size,),activation='relu'))
        model.add(Dense(32,activation='relu'))
        model.add(Dense(16,activation='relu'))
        model.add(Dense(self.action_size,activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1,verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)