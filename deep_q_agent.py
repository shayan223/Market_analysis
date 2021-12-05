
from dataset import market_graph_dataset

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class environment:
    '''#############################
        Currently just generates random numbers,
        eventually the "Environment State" will
        be replaced with aproximated graph values
        that are output by the respective graph CNN's
    #############################'''

    def __init__(self, data_root):
        self.cur_time_step = 0
        self.n_actions = 2
        #g1 = market_graph_dataset(csv_file=data_root+'candle_stick/labels.csv', root_dir=data_root+'\candle_stick')

        #TODO change to length of dataset once it is implemented
        self.max_steps = 1500

        #TODO update state to initialize with the get_state function
        self.state = 0


    def step(self, action):
        #returns reward and binary for whether the last step has been reached

        #reward nothing if price doesn't change
        reward = 0

        #TODO change price_change to read state label once data is implemented
        price_change = random.uniform(-1, 1)

        #reward buying before price increases
        #Punish when buying before price decreases
        #TODO Current assumption: 0 buy and 1 sell
        if price_change > 0 and action < 1:
            reward = 1
        if price_change < 0 and action < 1:
            reward = -1

        #reward for selling before price decreases
        #punishing for selling before price increases
        if price_change < 0 and action >= 1:
            reward = 1
        if price_change > 0 and action >= 1:
            reward = -1

        done = 0

        if(self.cur_time_step == self.max_steps):
            done = 1

        #update to the next time_step
        self.cur_time_step += 1

        return reward, done

    def get_state_internal(self, time_step):
        # TODO replace with evaluating current timestep images through pretrained CNN's to create state vector
        # Currently just 5 random "approximations"
        return torch.rand(5)

    def get_state(self):
        return self.get_state_internal(self.cur_time_step)

def aproximate_value(state):
    #TODO replace with evaluating current timestep images through pretrained CNN's to create state vector
    #Currently just 5 random "aproximations"
    return torch.rand(5)


'''
Based on the following tutorial 
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))




class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class NN_sig(nn.Module):

    #inputs: number of approximated features and/or CNN aproximator outputs
    #outputs: currently 2 outputs, buy and sell
    def __init__(self, inputs, outputs):
        super(NN_sig, self).__init__()

        self.Dense1 = nn.Linear(inputs, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.Dense2 = nn.Linear(16, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.Dense3 = nn.Linear(32,32)
        self.bn3 = nn.BatchNorm1d(32)
        self.out = nn.Linear(32, outputs)

    def forward(self, x):
        x = x.to(device)

        '''
        x = self.Dense1(x)
        x = torch.sigmoid(x)
        x = self.Dense2(x)
        x = torch.sigmoid(x)
        x = self.out(x)
        output = torch.sigmoid(x)
        return output'''
        x = F.relu(self.bn1(self.Dense1(x)))
        x = F.relu(self.bn2(self.Dense2(x)))
        x = F.relu(self.bn3(self.Dense3(x)))

        return self.out(x.view(x.size(0), -1))


REPLAY_MEMORY = 200
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
NUM_EPISODES = 50
env = environment('./fake_root')

# Get number of actions from gym action space
n_actions = env.n_actions

#Number of graphs being used in approximation
feature_count = 5

policy_net = NN_sig(feature_count,n_actions).to(device)
target_net = NN_sig(feature_count,n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(REPLAY_MEMORY)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    #Take normal action with chance 1-e, random action with chance e
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated



def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()





for i_episode in range(NUM_EPISODES):
    # Initialize the environment and state
    env = environment('./fake_root')

    state = env.get_state()
    next_state = env.get_state()

    for t in count():
        # Select and perform an action
        action = select_action(state)
        reward, done = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state = env.get_state()
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()

