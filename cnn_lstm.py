
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models


from dataset import market_graph_dataset

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQUENCE_LENGTH = 8
ENCODING_DIM = 16 #dimension of the encoding layer from each CNN
REPLAY_MEMORY = 64
BATCH_SIZE = 4
GAMMA = 0.5 # discount factor
EPS_START = 0.3
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 25
NUM_EPISODES = 2000
STEPS_PER_EP = 24
DATA_ROOT = './data/hourly/'
VALIDATION_SPLIT = .2
VALIDATION_EPISODES = 500


class environment:


    def __init__(self, data_root, validation_split,steps_per_ep,sequence_length):
        self.cur_time_step = 0
        self.n_actions = 2
        self.sequence_length = sequence_length
        self.g1 = market_graph_dataset(csv_file=data_root+'candle_stick/labels.csv', root_dir=data_root+'/candle_stick')
        self.g2 = market_graph_dataset(csv_file=data_root + 'movingAvg/labels.csv', root_dir=data_root + '/movingAvg')
        self.g3 = market_graph_dataset(csv_file=data_root + 'PandF/labels.csv', root_dir=data_root + '/PandF')
        self.g4 = market_graph_dataset(csv_file=data_root + 'price_line/labels.csv', root_dir=data_root + '/price_line')
        self.g5 = market_graph_dataset(csv_file=data_root + 'renko/labels.csv', root_dir=data_root + '/renko')


        self.total_steps = len(self.g1)
        #max time step before validation data cutoff
        #(updated to the correct value at the bottom of the constructor with set_data_split)
        self.max_steps = self.total_steps

        self. steps_per_ep = steps_per_ep

        self.state, self.current_label = self.get_state()

        self.offset_start = 0

        #Marks the start of the timesteps saved for validation
        self.validation_index = 0

        #split testing and validation time steps
        self.set_data_split(validation_split)

    def set_start_offset(self, offset):
        self.offset_start = offset
        self.cur_time_step = offset

    def set_data_split(self, validation_split):
        #all image datasets should be the same size always, as they are made of the same time series data
        dataset_size = len(self.g1)
        split = int(np.floor(validation_split * dataset_size))
        self.max_steps = self.total_steps - split
        self.validation_index = self.max_steps + 1
        print("Percent validation: ",validation_split)
        print("Training: t=0 to t=",self.max_steps,"Validation: t=",self.validation_index," to t=",self.total_steps)


    def step(self):

        done = 0

        #make sure to accomodate bounds of the sequence
        if(self.cur_time_step+self.sequence_length == self.max_steps):
            done = 1

        #update to the next time_step
        self.cur_time_step += 1
        #update state information
        self.state, self.current_label = self.get_state()

        #Stop training at the end of the episode
        if(self.cur_time_step == (self.offset_start+self.steps_per_ep)):
            done = 1


        return done

    def get_state_internal(self, time_step):

        sequence = []
        sequence_labels = []
        for i in range(self.sequence_length):
            img1 = torch.tensor(self.g1[time_step+i][0])
            img2 = torch.tensor(self.g2[time_step+i][0])
            img3 = torch.tensor(self.g3[time_step+i][0])
            img4 = torch.tensor(self.g4[time_step+i][0])
            img5 = torch.tensor(self.g5[time_step+i][0])
            time_step_label = self.g1[time_step+i][1]

            img_list = [img1, img2, img3, img4, img5]

            #Re-order dims to match (channel, dim2, dim1) format
            for i in range(len(img_list)):
                img_list[i] = img_list[i].unsqueeze(0).permute(0,3,1,2).type(torch.FloatTensor)

            element_tensor = torch.stack(img_list, dim=0)

            sequence.append(element_tensor)
            sequence_labels.append(time_step_label)

        #move the batch dimension to the first index
        state_tensor = torch.stack(sequence, dim=0).squeeze().unsqueeze(dim=0)

        return state_tensor, sequence_labels


    def get_state(self):
        return self.get_state_internal(self.cur_time_step)



#LSTM code based on the following tutorial https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# as well as this project combining CNN's and LSTM:
# https://github.com/pranoyr/cnn-lstm and https://github.com/pranoyr/cnn-lstm/blob/master/models/cnnlstm.py



class Ensemble(nn.Module):
    #combine all the CNN approximators and NN models
    def __init__(self, cnn1, cnn2, cnn3, cnn4, cnn5, num_classes):
        super(Ensemble, self).__init__()

        self.cnn1 = cnn1.to(device)
        self.cnn2 = cnn2.to(device)
        self.cnn3 = cnn3.to(device)
        self.cnn4 = cnn4.to(device)
        self.cnn5 = cnn5.to(device)

        self.cnn1.train()
        self.cnn2.train()
        self.cnn3.train()
        self.cnn4.train()
        self.cnn5.train()

        self.lstm = nn.LSTM(cnn1.fc.out_features*5, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, sequence):
        #this condition handles a batch of states versus a single state containing 5 images
        hidden = None
        for t in range(sequence.size(1)):
            xlist = sequence[:,t,:,:,:,:]#list of 3-dim images at given time step t

            x1 = xlist[:,0].to(device)
            x2 = xlist[:,1].to(device)
            x3 = xlist[:,2].to(device)
            x4 = xlist[:,3].to(device)
            x5 = xlist[:,4].to(device)

            x1 = self.cnn1(x1)
            x2 = self.cnn2(x2)
            x3 = self.cnn3(x3)
            x4 = self.cnn4(x4)
            x5 = self.cnn5(x5)

            x = torch.cat((x1,x2,x3,x4,x5),dim=1)

            out, hidden = self.lstm(x.unsqueeze(0), hidden)

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        #NOTE because we are using CrossEntropyLoss, we should NOT use softmax here, as it is applied in the loss function
        return x


def load_model(encoding_dim,model_path=None):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, encoding_dim)
    if(model_path is not None):
        model.load_state_dict(torch.load(model_path))
    #model.eval()

    return model




env = environment(DATA_ROOT, validation_split=VALIDATION_SPLIT,steps_per_ep=STEPS_PER_EP,sequence_length=SEQUENCE_LENGTH)

# Get number of actions from gym action space
n_actions = env.n_actions

#Number of graphs being used in approximation
feature_count = 5

#TODO Try this with fresh resnet models
#initialise ensembled models
#NOTE: switch encoding dim to 2 to use transfer-learned models in the models directory
cnn1 = load_model(ENCODING_DIM)
cnn2 = load_model(ENCODING_DIM)
cnn3 = load_model(ENCODING_DIM)
cnn4 = load_model(ENCODING_DIM)
cnn5 = load_model(ENCODING_DIM)


#Create ensemble
policy_net = Ensemble(cnn1, cnn2, cnn3, cnn4, cnn5,num_classes=n_actions).to(device)
target_net = Ensemble(cnn1, cnn2, cnn3, cnn4, cnn5,num_classes=n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=.01)
criterion = nn.CrossEntropyLoss()
steps_done = 0



episode_accuracy = []


#TODO Currently many-to-one prediction, consider training by predicting many-to-many
#TODO do this by returning all hidden states as sequential predictions and comparing to labels for every sequence step

print("##### Starting training ######")
for i_episode in range(NUM_EPISODES):

    print("Episode: ", i_episode)
    accuracy_sum = 0


    # Initialize the environment and state
    env = environment(DATA_ROOT,validation_split=VALIDATION_SPLIT,steps_per_ep=STEPS_PER_EP,sequence_length=SEQUENCE_LENGTH)

    #Start every episode at a random point in time
    start_t = random.randint(0,(env.max_steps-STEPS_PER_EP))
    print("Episode starting from t= ",start_t," to t= ",(start_t+STEPS_PER_EP))
    env.set_start_offset(start_t)

    state = env.state
    next_state = env.state

    print(policy_net(state))

    print("Training for ", STEPS_PER_EP, " steps.")
    #loop until the environment indicates the end of an episode
    for t in tqdm(count()):


        pred = policy_net(state)

        if(env.current_label[-1] < 0): label = torch.tensor([0]).to(device)
        else: label = torch.tensor([1]).to(device)

        #print(pred.max(1)[1].view(1, 1).item())
        #print(label.item())
        if(pred.max(1)[1].view(1, 1).item() == label.item()):
            accuracy_sum += 1

        loss = criterion(pred,label)

        # Select and perform an action
        done = env.step()


        # Observe new state
        if not done:
            next_state = env.state
        else:
            next_state = None


        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if done:
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    #compute and store average accuracy
    episode_accuracy.append((accuracy_sum/STEPS_PER_EP))

print('Complete')

# Plot and save loss
X_axis = list(range(NUM_EPISODES))
cur_plot = plt.figure()
plt.plot(X_axis, episode_accuracy, label='Training Accuracy')

#cur_plot.suptitle(str(model_name) + ' Loss: lr=' + str(lr))
plt.xlabel('Episode')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Average Training Accuracy: "+ str(sum(episode_accuracy)/len(episode_accuracy)))
# plt.show()
plt.savefig('./Training_base_cnn_lstm.png')

# clear plot for next iteration
plt.clf()



'''##################################################################'''
'''#################### Now evaluate the model ######################'''
'''##################################################################'''

episode_accuracy = []
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

print("##### Starting Validation ######")
for i_episode in range(VALIDATION_EPISODES):

    print("Episode: ", i_episode)
    accuracy_sum = 0


    # Initialize the environment and state
    env = environment(DATA_ROOT,validation_split=VALIDATION_SPLIT,steps_per_ep=STEPS_PER_EP,sequence_length=SEQUENCE_LENGTH)

    #Start every episode at a random point in time
    #Only this time we choose from the reserved validation indices
    start_t = random.randint(env.validation_index,(env.total_steps-STEPS_PER_EP))
    print("Episode starting from t= ",start_t," to t= ",(start_t+STEPS_PER_EP))
    env.set_start_offset(start_t)

    state = env.state
    next_state = env.state

    print(policy_net(state))

    print("Training for ", STEPS_PER_EP, " steps.")
    #loop until the environment indicates the end of an episode
    for t in tqdm(count()):


        pred = target_net(state)

        if(env.current_label[-1] < 0): label = torch.tensor([0]).to(device)
        else: label = torch.tensor([1]).to(device)

        #print(pred.max(1)[1].view(1, 1).item())
        #print(label.item())
        if(pred.max(1)[1].view(1, 1).item() == label.item()):
            accuracy_sum += 1

        # Select and perform an action
        done = env.step()


        # Observe new state
        if not done:
            next_state = env.state
        else:
            next_state = None


        # Move to the next state
        state = next_state


        if done:
            break


    #compute and store average accuracy
    episode_accuracy.append((accuracy_sum/STEPS_PER_EP))

print('Complete')

# Plot and save loss
X_axis = list(range(NUM_EPISODES))
cur_plot = plt.figure()
plt.plot(X_axis, episode_accuracy, label='Validation Accuracy')

#cur_plot.suptitle(str(model_name) + ' Loss: lr=' + str(lr))
plt.xlabel('Episode')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Average Validation Accuracy: "+ str(sum(episode_accuracy)/len(episode_accuracy)))
# plt.show()
plt.savefig('./Validation_base_cnn_lstm.png')

# clear plot for next iteration
plt.clf()