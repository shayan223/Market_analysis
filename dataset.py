import numpy as np
import pandas as pd
import torch
import os
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class market_graph_dataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.graphs = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.graphs)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.graphs.iloc[idx, -1])
        image = io.imread(img_name)
        percent_change = self.graphs.iloc[idx, 2]
        #time_step = np.array([time_step])

        sample = {'image': image, 'percent_change': percent_change}

        if self.transform:
            sample = self.transform(sample)

        return sample



def test():

    data = market_graph_dataset(csv_file='./data/daily/candle_stick/labels.csv', root_dir='./data/daily/candle_stick/')

    for i in range(len(data)):
        sample = data[i]

        print(sample['percent_change'])


#test()