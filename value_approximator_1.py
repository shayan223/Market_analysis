
from dataset import market_graph_dataset

import copy
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def train_resnet_approximator(model_name,label_csv,data_dir,out_dir,batch_size=8,validation_split=.2,epochs=25,lr=0.001):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    #TODO DOUBLE CHECK TO SEE IF I ACTUALLY NEED THE TRANSFORM
    data_transformer = transforms.Compose([transforms.Scale])
    data = market_graph_dataset(csv_file=label_csv, root_dir=data_dir)


    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    np.random.seed(int(time.time()))#time is used for random seeding can be set static for debug
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    dataset_sizes = {'train': len(train_indices), 'val': len(val_indices)}

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                    sampler=valid_sampler)

    #dataloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0)

    dataloaders = {'train': train_loader, 'val': validation_loader}

    '''training routine based on https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html'''
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

        train_acc = []
        val_acc = []
        train_loss = []
        val_loss = []

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:

                    #Labels are currently percent change in market value
                    #Convert them to binary, indicating only weather the price went up or down

                    labels[labels < 0] = 0 #all negative values to 0
                    labels[labels > 0] = 1 #all positive values to 1

                    #reorder input to match format: (Batch_size, channels, dim1, dim2)
                    inputs = inputs.permute(0,3,1,2)

                    #Send data to GPU
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):

                        #cast input as float tensor (it starts as byteTensor)
                        inputs = inputs.float()
                        #Labels need to be converted from double to long
                        labels = labels.long()

                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)

                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                #Record per-epoch accuracy and loss for both training and validation
                if phase == 'train':
                    train_acc.append(epoch_acc.item())
                    train_loss.append(epoch_loss)
                else:
                    val_acc.append(epoch_acc.item())
                    val_loss.append(epoch_loss)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))



        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, train_acc, val_acc, train_loss, val_loss


    '''#########################################################'''

    #TODO should I train the whole network? (resnet18 is pretty tiny) or transfer by only changing final layer
    #alternatively I could use a bigger resnet and only tune the output layer

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    '''#########################################################'''

    model_ft, train_acc, val_acc, train_loss, val_loss= train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=epochs)


    #Plot and save loss
    X_axis = list(range(epochs))
    cur_plot = plt.figure()
    plt.plot(X_axis, train_loss, label='Training Loss')
    plt.plot(X_axis, val_loss, label='Validation Loss')
    cur_plot.suptitle(str(model_name)+' Loss: lr=' + str(lr))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig(str(out_dir)+str(model_name)+'_loss.png')

    # clear plot for next iteration
    plt.clf()

    #Plot and save accuracy
    X_axis = list(range(epochs))
    cur_plot = plt.figure()
    plt.plot(X_axis, train_acc, label='Training Accuracy')
    plt.plot(X_axis, val_acc, label='Validation Accuracy')
    cur_plot.suptitle(str(model_name)+' Accuracy: lr=' + str(lr))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.show()
    plt.savefig(str(out_dir)+str(model_name)+'_acc.png')

    # clear plot for next iteration
    plt.clf()

    return model_ft