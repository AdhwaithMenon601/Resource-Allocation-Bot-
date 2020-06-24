import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import style
import gym
from gym import utils
from gym.envs.toy_text import discrete

"""
The actions are as follows -
0 : Add to M1
1 : Add to M2
2 : Add to M3
3 : Do nothing 
"""

#Render is not required for the given function
class ResourceAlloc(gym.Env):
    def __init__(self, m1, m2, m3, n):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.n = n
        self.cur_state = (0,0,0)
        #self.state_space = []

        """
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    self.state_space.append((i,j,k))
        """


    def find_min_time(self, x, y, z):
        return (self.m1 * x) + (self.m2 * y) + (self.m3 * z)

    def reset(self):
        #To reset the evnironment
        #Set the current state to (0,0,0) and rewards to normal
        self.cur_state = (0,0,0)

        return self.cur_state

    def step(self, action):
        #Take in the given action provided given state
        done = False
        (x,y,z) = self.cur_state
        #print('The x y z are ',x,y,z)

        #print('The values are ',x,y,z)
        reward = self.find_min_time(x,y,z)
        my_sum = x + y + z
        #print('The action is ',action)
        #print('The sum is ',my_sum)
        if (my_sum == (self.n - 1)):
            reward = 1/reward
            reward *= 100
            done = True
        else :
            reward = -1           #Our basic reward function
            done = False

        #Changing the values of the states
        #Will try and implement a smarter version of this
        if (action == 0):
            x += 1
        elif (action == 1):
            y += 1
        elif (action == 2):
            z += 1


        if (x > (self.n+1)) or (y > (self.n+1)) or (z > (self.n+1)):
            done = False
            x = 0
            y = 0
            z = 0
            
        next_state = (x,y,z)
        self.cur_state = (x,y,z)
        #Once again checking for equal sum
        #Only in case it does not work
        new_sum = x + y + z
        if (new_sum != self.n):
            done = False

        return next_state, reward, done