from torch.utils.data import DataLoader,Dataset,random_split
import os
import csv
import random
import numpy as np
import torch
import itertools
import glob

class Dydata(Dataset):
    def __init__(self, root, train=True, ratio=0.8):
        self.train=train
        self.path = os.listdir(root)
        self.path = [os.path.join(root, self.path[i]) for i in range(len(self.path))]
        self.path = [glob.glob(os.path.join(self.path[i], "*.npy")) for i in range(len(self.path))]
        self.path = list(itertools.chain.from_iterable(self.path))




    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):

        data = np.load(self.path[index])
        data = torch.tensor(data)
        label = int(self.path[index].split("\\")[-2])
        label = torch.tensor(label)

        return data[:,:,:3], label

if __name__ == '__main__':
    import torch
    from torch.utils.data import Subset, DataLoader
    import numpy as np

    dataset = Dydata(r"E:\代码\dataprocess\data_own\data_processed2\test")
    torch.manual_seed(1234)
    train_ratio = 0.8
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    rng = torch.Generator()
    rng.manual_seed(1234)

    indices_x = []
    indices_y = []
    i = 0
    j = i+1;
    while i<len(dataset):
        indices_x.append(i)
        indices_y.append(i+1)
        i+=2

    query_dataset = Subset(dataset,indices_x)
    gallery_dataset = Subset(dataset,indices_y)







