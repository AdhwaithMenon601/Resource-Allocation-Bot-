import sys
import numpy as np
import gym
import envs
from dqn_keras import Agent
from Robot import Robot

#TBD m1 , m2 , m3 are ratios of different machines 
#TBD Print exact order of the robots
def robot_order(final_state, num_bots, num_machines):
    #Function to print exact order
    is_finished = False
    state_size = num_machines * num_bots
    is_machine_busy = np.zeros(num_machines)
    sub_final_states = [final_state[x:x+num_machines] for x in range(0,state_size,num_machines)]
    flag = 0
    #print(sub_final_states)

    while(not is_finished):
        for i in range(num_bots):
            for j in range(num_machines):
                if (is_machine_busy[j] == 0) and (sub_final_states[i][j] > 0):
                    print("Robot {} takes part to machine {} ".format(i+1, j+1))
                    sub_final_states[i][j] -= 1
                    is_machine_busy[j] = 1
                    break
        print("Waiting till machines finish ...")
        for k in range(num_machines):
            is_machine_busy[k] = 0
        
        for i in range(num_bots):
            for j in sub_final_states[i]:
                if (j != 0):
                    flag = 1
                    break
        #Checking within the loop
        if (flag == 0):
            is_finished = True
            break
        flag = 0

#Above function will iterate through the entire final_state
#Then it will one by one reduce the parts for each given robot and print the steps
#As it loops , it will access each of the robots parts
def create_machine(m1, m2, m3, n, num_bots, time_per_bot):
    #Create the environment from gym
    (a,b,c,d) = (m1,m2,m3,n)
    env = gym.make('ResourceAlloc-v1',m1=a,m2=b,m3=c,n=d)
    state_size = 3     #Should be no of dimensions
    action_size = 4

    #States will be for each robot respectively
    #Actions are 4 for each robot=
    #Move to M1 , Move to M2 , Move to M3 , Stay Idle (Per robot)
    robot_state_size = num_bots * state_size
    robot_action_size = num_bots * action_size

    n_episodes = 25
    batch_size = 32

    agent = Agent(state_size, action_size)

    final_state = []
    max_score = 0

    done = False
    for e in range(n_episodes):

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(10000):

            #Mostly won't need to render the environment
            #env.render()
            action = agent.act(state)

            next_state, reward, done = env.step(action)

            #reward = reward if not done else -100
            if (done):
                _ = True
            else:
                #Reward is -10 as range for reward function is around (0,10)
                reward = -10

            temp_state = next_state
            temp_state = np.reshape(temp_state, [state_size,])
            next_state = np.reshape(next_state, [1, state_size])

            #print(next_state)
            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                if (reward > max_score):
                    max_score = reward
                    final_state = temp_state
                print("episode: {}/{}, score: {}".format(e,n_episodes, time))
                break


        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if e % 50 == 0:
            agent.save("resource_weights_" + '{:04d}'.format(e) + ".hdf5")

    #To get the predicted value , we must append all states to an array
    #and from that choose the best
    (x,y,z) = final_state
    print("The final sequence is hence M1:{} parts M2:{} parts M3:{} parts ".format(x,y,z))
    print("The final time for processing alone is ",max(a * x, b * y, c * z))
    proc_time = max(a * x, b * y, c * z)
    
    #From here we start allocation of robots towards the respective machines
    #Here we will assign the robot class and also create a neural network of the same
    #This network will be able to pinpoint the correct sequence of bots to be sent
    robot = Robot(num_bots, final_state, time_per_bot)
    new_agent = Agent(robot_state_size, robot_action_size)

    max_robot_score = 0

    n_episodes = 10
    done = False
    robot_final_state = []
    for e in range(n_episodes):
        state = robot.reset()
        state = np.reshape(state, [1, robot_state_size])

        for time in range(10000):

            #Mostly won't need to render the environment
            action = new_agent.act(state)

            next_state, reward, done = robot.step(action)

            if (done):
                _ = True
            else:
                #Reward is -10 as range for reward function is around (0,10)
                reward = -10

            temp_state = next_state
            next_state = np.reshape(next_state, [1, robot_state_size])


            new_agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                if (reward > max_robot_score):
                    max_robot_score = reward
                    robot_final_state = temp_state
                print("episode: {}/{}, score: {}".format(e,n_episodes, time))
                break


        if len(new_agent.memory) > batch_size:
            new_agent.replay(batch_size)

        if e % 50 == 0:
            new_agent.save("robot_weights_" + '{:04d}'.format(e) + ".hdf5")
        #Now to print out the final sequence for each robot for each part
        #We shall use a loop to iterate through the final state
        #Each robot gives to 3 machines
    bot_no = 1
    bot_time = []
    k = 0
    final_bot_time = 0

    #This current code is only a test code for three machines
    #The code will be made to accept a generalised input of machines
    #The only difference that needs to be done is simply to extract the values from the array
    #(a,b,c,d,e,f) = robot_final_state
    if (num_bots >= 3):
        print("Each bot can be assigned a specific machine in this case")
        k = 1
        final_bot_time = time_per_bot
    else:
        my_time = 0
        seq_time = [robot_final_state[x:x+3] for x in range(0,robot_state_size,3)]
        #Iterating through the loop per machine
        for i in range(num_bots):
            print("Following is for Robot {}".format(i+1))
            print("Parts to be delivered to M1 {} and M2 {} and M3 {}".format(seq_time[i][0],seq_time[i][1],seq_time[i][2]))
            for j in range(state_size):
                my_time += (seq_time[i][j] * time_per_bot)
            bot_time.append(my_time) 

    if (k == 0):
        final_bot_time = max(bot_time)
        #The create_machine function is now finished
        #We have custom made it per Part Type and for each machine required quantity
        #The time will be calculated later (As in total time from end to end)
    total_time = final_bot_time + proc_time
    print("The total time for finishing said type products is {}".format(total_time))
    #TBD we need a function to print exact order of the robots
    robot_order(robot_final_state, num_bots, num_machines=3)



#This is the main function
def main():
    print('The following is for machines')
    total = 10
    create_machine(4, 5, 4, 3, 2, 1)
    create_machine(2, 4, 3, 5, 2, 1)
    create_machine(1, 7, 3, 2, 2, 1)

if __name__ == '__main__':
    main()