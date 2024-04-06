# Generator synthetic data 
# Made them to a Pytorch Dataset 

from torch.utils.data import Dataset, DataLoader
import torch
from GANModels import *
import numpy as np
import os
from dataLoader import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Synthetic_Dataset(Dataset):
    def __init__(self, 
                 model_path='./pre-trained-models/checkpoint',
                 sample_size=None,
                 test_proportion=0.2,
                 dataset=None,
                 verbose=False,
                 **kwargs
                 ):
        #Generate Running Data
        gen_net = Generator(**kwargs)
        checkpoint = torch.load(model_path, map_location=device)
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        
        
        #Generate synthetic data; label is 0
        if dataset is None: dataset = load_dataset(data_mode='test', sample_size=sample_size, test_porportion=test_proportion)
        conditions = torch.from_numpy(dataset.X_train).type(torch.float)
        self.conditions = np.reshape(conditions, (conditions.shape[0], -1))
        z = torch.FloatTensor(np.random.normal(0, 1, (self.conditions.shape[0], 100)))
        self.syn = gen_net(z, self.conditions)
        self.syn = self.syn.detach().numpy()
        
        # self.combined_train_data = np.concatenate((self.syn_running, self.syn_running), axis=0)
        # print(self.combined_train_data.shape)
        # self.combined_train_data = np.concatenate((self.syn_running, self.syn_jumping), axis=0)
        # self.combined_train_label = np.concatenate((self.running_label, self.jumping_label), axis=0)
        # self.combined_train_label = self.combined_train_label.reshape(self.combined_train_label.shape[0], 1)
        
        if verbose:
            print(self.conditions.shape)
            print(self.syn.shape)
        
    
    def __len__(self):
        return self.conditions.shape[0]
    
    def __getitem__(self, idx):
        # return self.combined_train_data[idx], self.combined_train_label[idx]
        return self.conditions[idx], self.syn[idx]


if __name__ == '__main__':
    from torch.utils import data

    syn_data = Synthetic_Dataset()
    syn_data_loader = data.DataLoader(syn_data, batch_size=1, num_workers=1, shuffle=True)
    syn = []

    for i, (syn_sig, label) in enumerate(syn_data_loader):
        syn_sig = syn_sig.cpu().detach().numpy()
        sig = syn_sig.reshape(syn_sig.shape[1], syn_sig.shape[3])
        syn.append(sig)

    syn = np.array(syn)
    print(syn.shape)