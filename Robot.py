#Class for each of the individual bots
#Will be called from the run.py function
import os
import sys
import numpy as np
import random

#In our Robot , we shall have the following actions -
#We have a total of num_bots * 4 
#We put 4 as we have 4 types of operations - 
#Move robot to M1 , Move robot to M2 , Move robot to M3 , Stay robot idle
#The latter will rarely be used
#Once again we shall try and use the same neural network to build in 'run.py'
class Robot(object):
    def __init__(self, num_bots, final_state, time_per_bot):
        self.n = num_bots
        self.time_per_bot = time_per_bot
        self.final_state = final_state
        #We initialise this as a 1D array with numpy , it will be lengthwise.
        #For the current case , since we have 3 machines we shall multiply by 3
        #For a general scenario we shall input the number of machines also
        self.cur_state = np.zeros(num_bots * 3,dtype=int)
    
    #Given function is for resetting it at the end
    def reset(self):
        self.cur_state = np.zeros(self.n * 3,dtype=int)
        return np.array(self.cur_state)
    
    def find_min_time(self):
        sum = 0
        for i in self.cur_state:
            sum += (i * self.time_per_bot)
        return sum
    
    #Given function is similar to 'step' in the ResourceAlloc
    def step(self, action):
        done = True
        #The reward will be the inverse of the following
        reward = self.find_min_time()
        if (reward == 0):
            reward = -1

        #We shall use a similar reward function as used in the Resource Allocation
        #Once again , this is only for 3 machines 
        #For any number of machines , we shall have to use a List
        #For now , we shall limit to 3 machines at most
        flag = 1
        #For iterating in the loop
        total_parts_req = self.n * 3
        for i in range(3):
            my_sum = 0
            for j in range(i,total_parts_req,3):
                my_sum += self.cur_state[j]
            #Checking for each case 
            if (my_sum != self.final_state[i]):
                flag = 0
                done = False
                reward = -30
        
        if (flag == 0):
            reward = -10
        elif (flag == 1):
            reward = (1/reward)
            reward *= 10
            done = True
        
        (x,y,z) = self.final_state
        #Now we check for the required action
        #Again , due to convenience I have considered only 3 machines 
        #On addition of more machines we cannot use the following if-then-else statement
        if (action == 0):
            if (self.cur_state[0] >= x):
                self.cur_state[0] = 0
                done = False
            else:
                self.cur_state[0] += 1
        elif (action == 1):
            if (self.cur_state[1] >= y):
                self.cur_state[1] = 0
                done = False
            else:
                self.cur_state[1] += 1
        elif (action == 2):
            if (self.cur_state[2] >= z):
                self.cur_state[2] = 0
                done = False
            else:
                self.cur_state[2] += 1
        elif (action == 4):
            if (self.cur_state[3] >= x):
                self.cur_state[3] = 0
                done = False
            else:
                self.cur_state[3] += 1
        elif (action == 5):
            if (self.cur_state[4] >= y):
                self.cur_state[4] = 0
                done = False
            else:
                self.cur_state[4] += 1
        elif (action == 6):
            if (self.cur_state[5] >= z):
                self.cur_state[5] = 0
                done = False
            else:
                self.cur_state[5] += 1

        #This part is an extra function that once again checks if total is crossed
        #This needed to be added , as we need to check from start also due to reward
        for i in range(3):
            my_sum = 0
            for j in range(i,total_parts_req,3):
                my_sum += self.cur_state[j]
            #Checking for each case 
            if (my_sum != self.final_state[i]):
                flag = 0
                done = False
                reward = -30

        next_state = self.cur_state
        next_state = np.array(next_state)

        return next_state, reward, done        
                



