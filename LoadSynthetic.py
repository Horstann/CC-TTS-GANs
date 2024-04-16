# Generator synthetic data 
# Made them to a Pytorch Dataset 

from torch.utils.data import Dataset, DataLoader
import torch
from GANModels import *
import numpy as np
import os
from dataLoader import *
from functions import to_price_paths
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Synthetic_Dataset(Dataset):
    def __init__(
            self, 
            model_path='./pre-trained-models/checkpoint',
            sample_size=None,
            test_proportion=0.2,
            dataset=None,
            verbose=False,
            var=1,
            n=10,
            **kwargs
        ):
        # Generate Running Data
        self.gen_net = Generator(**kwargs)
        checkpoint = torch.load(model_path, map_location=device)
        self.gen_net.load_state_dict(checkpoint['gen_state_dict'])
        self.sim_var = var
        
        # Generate synthetic data; label is 0
        if dataset is None: self.dataset = load_dataset(data_mode='test', sample_size=sample_size, test_porportion=test_proportion)
        else: self.dataset = dataset
        self.conditions = np.repeat(self.dataset.X_test, n, axis=0)
        self.conditions = torch.from_numpy(self.conditions).type(torch.float)
        z = torch.FloatTensor(np.random.normal(0, 1, (self.conditions.shape[0], 100)))
        self.sims = self.gen_net(z, self.conditions).detach().numpy()

        if verbose:
            print(self.conditions.shape)
            print(self.sims.shape)
    
    def __len__(self):
        return self.conditions.shape[0]
    
    def __getitem__(self, idx):
        # return self.combined_train_data[idx], self.combined_train_label[idx]
        return self.conditions[idx], self.sims[idx]
    
    def get_sims(self, date, n=100):
        condition = [[list(self.dataset.df[self.dataset.condition_names].loc[date])]]
        conditions = torch.from_numpy(np.array([condition for i in range(n)])).type(torch.float)
        z = torch.FloatTensor(np.random.lognormal(0, self.sim_var, (n, 100)))
        sims = self.gen_net(z, conditions).detach().numpy()
        return sims

if __name__ == '__main__':
    from torch.utils import data

    syn_data = Synthetic_Dataset(model_path='./logs/latest/Model/checkpoint', seq_len=42, conditions_dim=2)
    syn_dataloader = data.DataLoader(syn_data, batch_size=1, num_workers=1, shuffle=True)

    # syn_sims = []
    # syn_paths = []
    # syn_conds = []

    # for i, (cond, sim) in enumerate(syn_dataloader):
    #     sim = sim.cpu().detach().numpy()
    #     sim = sim.reshape(sim.shape[1], sim.shape[3])
    #     syn_sims.append(sim)
    #     cond = cond.cpu().detach().numpy()
    #     syn_conds.append(cond)
        
    # syn_sims = np.array(syn_sims)
    # syn_paths = to_price_paths(syn_sims)
    # syn_conds = np.array(syn_conds)
    # print(syn_sims.shape)
    # print(syn_paths.shape)
    # print(syn_conds.shape)

    sims = syn_data.get_sims(date='2023-09-05')
    print(sims)