"""
Additional References  
 If you use the dataset and/or code, please cite this paper (downloadable from [here](http://www.mdpi.com/2076-3417/7/10/1101/html))

Author:  Lee B. Hinkle, IMICS Lab, Texas State University, 2021

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
"""

import os
import shutil #https://docs.python.org/3/library/shutil.html
from shutil import unpack_archive # to unzip
#from shutil import make_archive # to create zip for storage
import requests #for downloading zip file
from scipy import io #for loadmat, matlab conversion
import pandas as pd
import numpy as np
import yfinance as yf
import random
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt # for plotting - pandas uses matplotlib
from tabulate import tabulate # for verbose tables
#from tensorflow.keras.utils import to_categorical # for one-hot encoding
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class load_dataset(Dataset):
    def __init__(
        self,
        sample_size=None,
        batch_size=8,
        test_porportion=0.2,
        verbose=False,
        data_mode='train', augment_times=None
    ):
        self.sample_size = sample_size
        self.test_proportion = test_porportion
        self.input_size = 1
        self.output_size = 42
        self.verbose = verbose
        self.data_mode = data_mode
        self.scaler = None

        start_date = '1999-01-01'
        
        # Get & preprocess data
        close = yf.download('SPY', start=start_date)['Adj Close']
        close.index = pd.to_datetime(close.index)
        vix = yf.Ticker('^VIX').history(start=start_date)['Close']
        vix.index = pd.to_datetime(vix.index.tz_convert(None).normalize())
        self.df = pd.DataFrame(index=close.index)
        self.df['close'] = close
        self.df['logreturns'] = np.log(close).diff()
        self.df['vix'] = vix
        # Incorporate BBG data
        ivol = pd.read_csv('data/ivol.csv', index_col='Date')['ivol']
        ivol.index = pd.to_datetime(ivol.index, format='%d/%m/%Y')
        pc_ratio = pd.read_csv('data/putcall_ratio.csv', index_col='Date')['putcall_ratio']
        pc_ratio.index = pd.to_datetime(pc_ratio.index, format='%d/%m/%Y')
        self.df['ivol'] = ivol
        self.df['pc_ratio'] = pc_ratio
        # Transform/normalise conditions
        self.df, self.condition_names = self.transform_conditions(self.df)
        self.df.dropna(inplace=True)

        # Extract variables from self.df
        self.logreturns = np.array(self.df['logreturns'])
        self.conditions = np.concatenate([np.reshape(np.array(self.df[col]), (-1,1)) for col in self.condition_names], axis=1)

        # Sample from data to create new train & test sets
        total_sample_pts = len(self.logreturns) - (self.input_size+self.output_size) + 1
        if sample_size is None:
            sample_size = total_sample_pts
            step_size = 1
        else:
            assert sample_size <= total_sample_pts
            step_size = total_sample_pts / (sample_size - 1)
        sample_pts = np.array([min(int(round(step_size * i)), total_sample_pts-1) for i in range(sample_size)])
        self.X = np.array([np.reshape(self.conditions[pt:pt+self.input_size,:], (1,1,-1)) for pt in sample_pts])
        self.Y = np.array([np.reshape(self.logreturns[pt+self.input_size:pt+self.input_size+self.output_size], (1,1,-1)) for pt in sample_pts])
        self.X = self.X.astype(np.float32)
        self.Y = self.Y.astype(np.float32)

        # self.normalise()

        # Split train/test sets
        if batch_size is not None:
            # If batch_size given, make train_size divisible by batch_size
            num_train_batches = round(len(self.X) * (1-self.test_proportion) / batch_size)
            train_size = num_train_batches * batch_size
            test_size = len(self.X) - train_size
        else:
            test_size = round(len(self.X) * self.test_proportion)
        self.X_train, self.X_test, self.Y_train, self.Y_test = self.X[:-test_size,:,:,:], self.X[-test_size:,:,:,:], self.Y[:-test_size,:,:,:], self.Y[-test_size:,:,:,:]
            
        print(f"X_train's shape is {self.X_train.shape}, X_test's shape is {self.X_test.shape}")
        print(f"y_train's label shape is {self.Y_train.shape}, y_test's label shape is {self.Y_test.shape}")

    def zscore(self, series, window, keep_first_window=True):
        series = np.log(series.copy())
        r = series.rolling(window=window)
        m = r.mean().shift(1)
        s = r.std(ddof=1).shift(1)
        z = (series-m)/s
        if keep_first_window:
            first_window = series[:window]
            z[:window] = (first_window-first_window.mean())/first_window.std(ddof=1)
        return z
    def rsi(self, logreturns, window):
        pos_returns = logreturns.copy()
        neg_returns = logreturns.copy()
        pos_returns[pos_returns<0] = 0
        neg_returns[neg_returns>0] = 0
        pos_avg = pos_returns.rolling(window).mean()
        neg_avg = neg_returns.rolling(window).mean().abs()
        return 100 * pos_avg / (pos_avg + neg_avg)    

    def transform_conditions(self, df):
        df['rsi11d'] = self.rsi(df['logreturns'], window=11)
        df['rsi11d_z252d'] = self.zscore(df['rsi11d'], window=252)
        # df['vix_z252d'] = self.zscore(df['vix'], window=252)
        df['vix_z42d'] = self.zscore(df['vix'], window=42)
        df['ivol_z252d'] = self.zscore(df['ivol'], window=252)
        # df['ivol_z42d'] = self.zscore(df['ivol'], window=42)
        df['pc_ratio_z252d'] = self.zscore(df['pc_ratio'], window=252)
        # df['pc_ratio_z42d'] = self.zscore(df['pc_ratio'], window=42)
        # condition_names = ['rsi11d_z252d', 'vix_z252d', 'vix_z42d', 'ivol_z252d', 'ivol_z42d', 'pc_ratio_z252d', 'pc_ratio_z42d']
        condition_names = ['rsi11d_z252d', 'vix_z42d', 'ivol_z252d', 'pc_ratio_z252d']
        return df, condition_names

    # def normalise(self, rolling_window_size=None, keep_first_window_size=252):
    #     self.X_init = self.X.copy()
    #     scaler = StandardScaler()

    #     # For the first min_window_size data points, fit_transform as usual so as not to lose any datapoints
    #     assert keep_first_window_size > 1
    #     self.X[:keep_first_window_size,0,0,:] = scaler.fit_transform(self.X_init[:keep_first_window_size,0,0,:])
    #     # For data points after, use expanding or rolling window
    #     if rolling_window_size is not None:
    #         # Rolling window
    #         assert rolling_window_size <= keep_first_window_size
    #         for i in range(keep_first_window_size, self.X_init.shape[0]):
    #             scaler.fit(self.X_init[i-rolling_window_size:i,0,0,:])
    #             self.X[i:i+1,0,0,:] = scaler.transform(self.X_init[i:i+1,0,0,:])
    #     else:
    #         # Expanding window
    #         for i in range(keep_first_window_size, self.X_init.shape[0]):
    #             scaler.fit(self.X_init[:i,0,0,:])
    #             self.X[i:i+1,0,0,:] = scaler.transform(self.X_init[i:i+1,0,0,:])
    #     self.scaler = scaler
    #     # So that X & Y's lengths match
    #     self.Y = self.Y[-len(self.X):]
    
    def __len__(self):
        if self.data_mode == 'train': return len(self.Y_train)
        else: return len(self.Y_test)
        
    def __getitem__(self, idx):
        if self.data_mode == 'train': return self.X_train[idx], self.Y_train[idx], 
        else: return self.X_test[idx], self.Y_test[idx]
            
if __name__ == '__main__':
    from torch.utils import data
    os.chdir(r'C:\Users\Horstann\Documents\NTU URECA\tts-gan-main')

    batch_size = 8
    num_workers = 8
    augment_times = None

    train_set = load_dataset(data_mode='Train', augment_times=augment_times, batch_size=batch_size)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    # test_set = load_dataset(incl_val_group=False, data_mode='Test', single_class=True)
    # test_loader = data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
