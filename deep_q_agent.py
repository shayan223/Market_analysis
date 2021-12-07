
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
from torchvision import models

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
        self.g1 = market_graph_dataset(csv_file=data_root+'candle_stick/labels.csv', root_dir=data_root+'/candle_stick')
        self.g2 = market_graph_dataset(csv_file=data_root + 'movingAvg/labels.csv', root_dir=data_root + '/movingAvg')
        self.g3 = market_graph_dataset(csv_file=data_root + 'PandF/labels.csv', root_dir=data_root + '/PandF')
        self.g4 = market_graph_dataset(csv_file=data_root + 'price_line/labels.csv', root_dir=data_root + '/price_line')
        self.g5 = market_graph_dataset(csv_file=data_root + 'renko/labels.csv', root_dir=data_root + '/renko')

        self.ap1 = load_model('./models/candle_stick/candle_stick.pt')
        self.ap2 = load_model('./models/movingAvg/movingAvg.pt')
        self.ap3 = load_model('./models/PandF/PandF.pt')
        self.ap4 = load_model('./models/price_line/price_line.pt')
        self.ap5 = load_model('./models/renko/renko.pt')

        self.max_steps = len(self.g1)

        self.state, self.current_label = self.get_state()

        self.state_approximation = self.aproximate_values()


    def step(self, action):
        #returns reward and binary for whether the last step has been reached

        #reward nothing if price doesn't change
        reward = 0

        #TODO change price_change to read state label once data is implemented
        price_change = self.current_label

        #reward buying before price increases
        #Punish when buying before price decreases
        #TODO Current assumption: 0 sell and 1 buy
        if price_change > 0 and action >= 1:
            reward = 1
        if price_change < 0 and action >= 1:
            reward = -1

        #reward for selling before price decreases
        #punishing for selling before price increases
        if price_change < 0 and action < 1:
            reward = 1
        if price_change > 0 and action < 1:
            reward = -1

        done = 0

        if(self.cur_time_step == self.max_steps):
            done = 1

        #update to the next time_step
        self.cur_time_step += 1
        #update state information
        self.state, self.current_label = self.get_state()
        self.state_approximation = self.aproximate_values()

        '''######### IF STATEMENT FOR TESTING, CUTS TRAINING SHORT ########'''
        #TODO consider starting at different points in the time line
        if(self.cur_time_step == STEPS_PER_EP):
            done = 1
        '''################################################################'''
        return reward, done

    def get_state_internal(self, time_step):

        img1 = torch.tensor(self.g1[time_step][0])
        img2 = torch.tensor(self.g2[time_step][0])
        img3 = torch.tensor(self.g3[time_step][0])
        img4 = torch.tensor(self.g4[time_step][0])
        img5 = torch.tensor(self.g5[time_step][0])
        time_step_label = self.g1[time_step][1]

        img_list = [img1, img2, img3, img4, img5]

        #Re-order dims to match (channel, dim2, dim1) format
        for i in range(len(img_list)):
            img_list[i] = img_list[i].unsqueeze(0).permute(0,3,1,2).type(torch.FloatTensor)

        return img_list, time_step_label

        # Currently just 5 random "approximations"
        #return torch.rand(5)

    def get_state(self):
        return self.get_state_internal(self.cur_time_step)

    def aproximate_values(self):

        #NOTE: squeeze() simply removes empty (extra in this case) dimensions
        output_vector = [
        self.ap1(self.state[0]).squeeze(),
        self.ap2(self.state[1]).squeeze(),
        self.ap3(self.state[2]).squeeze(),
        self.ap4(self.state[3]).squeeze(),
        self.ap5(self.state[4]).squeeze()
            ]

        #Take output <sell_confidence, buy_confidence> and convert it to scalar positive (buy confidence)
        # or scalar negative (sell confidence)
        approximation_vector = []
        for i in range(len(output_vector)):
            #to_buy is 0 for sell, 1 for buy
            to_buy = torch.argmax(output_vector[i])
            output_vector[i]
            result = output_vector[i][to_buy]
            #TODO FIND A BETTER WAY TO HANDLE THIS SCENERIO
            # in cases where the highest confidence is still negative, set to 0
            if(result < 0):
                result = 0

            #Now convert that confidence value to positive if buy, negative if sell
            if(to_buy == 0):
                result = result * -1

            approximation_vector.append(result)


        approximation_vector = torch.tensor(approximation_vector)

        return approximation_vector



def load_model(model_path):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model

'''
Based on the following tutorial 
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


REPLAY_MEMORY = 8
BATCH_SIZE = 4
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
NUM_EPISODES = 40
STEPS_PER_EP = 150
DATA_ROOT = './data/daily/'
env = environment(DATA_ROOT)


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

        x = self.Dense1(x)
        #x = x.unsqueeze(0)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.Dense2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.Dense3(x)
        x = self.bn3(x)
        x = F.relu(x)

        return self.out(x.view(x.size(0), -1))




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
        '''
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)'''
        policy_net.eval()

        val = policy_net(state.unsqueeze(0)).max(1)[1].view(1, 1)
        policy_net.train()
        return val
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    #plt.pause(0.001)  # pause a bit so that plots are updated



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
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])


    state_batch = torch.stack(batch.state,dim=0)#.unsqueeze(1)
    action_batch = torch.stack(batch.action,dim=0).flatten(1)
    reward_batch = torch.stack(batch.reward,dim=0)#.unsqueeze(1)
    '''
    state_batch = batch.state
    action_batch = batch.action
    reward_batch = batch.reward
    '''

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
    # changed from the below to get * and + operations to behave element-wise
    #expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    next_state_values = next_state_values.squeeze()*GAMMA
    reward_batch = reward_batch.squeeze()

    expected_state_action_values = torch.add(next_state_values,reward_batch).unsqueeze(1)

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



episode_rewards = []

for i_episode in range(NUM_EPISODES):

    print("Episode: ", i_episode)
    reward_sum = 0


    # Initialize the environment and state
    env = environment(DATA_ROOT)

    state = env.state_approximation
    next_state = env.state_approximation

    for t in count():
        # Select and perform an action
        action = select_action(state)
        reward, done = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        #print(reward)
        reward_sum += reward.item()

        # Observe new state
        if not done:
            next_state = env.state_approximation
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        print('Step: ', t)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            #plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    episode_rewards.append(reward_sum)

print('Complete')

# Plot and save loss
X_axis = list(range(NUM_EPISODES))
cur_plot = plt.figure()
plt.plot(X_axis, episode_rewards, label='Training Reward')

#cur_plot.suptitle(str(model_name) + ' Loss: lr=' + str(lr))
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
# plt.show()
plt.savefig('./Deep_Q_Results.png')

# clear plot for next iteration
plt.clf()


