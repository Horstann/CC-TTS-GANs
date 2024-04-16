# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import collections
import logging
import math
import os
import time
from datetime import datetime
import dateutil.tz
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from scipy.integrate import nquad
from scipy.linalg import sqrtm
import pickle
from tqdm import tqdm

from typing import Union, Optional, List, Tuple, Text, BinaryIO
import pathlib
import torch
import math
import warnings
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor

@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    **kwargs
) -> torch.Tensor:
    """
    Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    Returns:
        grid (Tensor): the tensor containing grid of images.
    Example:
        See this notebook
        `here <https://github.com/pytorch/vision/blob/master/examples/python/visualization_utils.ipynb>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
    

def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))

def plot_2axes(df, y1, y2):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=df.index, y=df[y1], name=y1),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df[y2], name=y2),
        secondary_y=True,
    )
    # Set y-axes titles
    fig.update_yaxes(title_text=y1, secondary_y=False)
    fig.update_yaxes(title_text=y2, secondary_y=True)
    fig.show()

class RunningStats:
    def __init__(self, WIN_SIZE):
        self.mean = 0
        self.run_var = 0
        self.WIN_SIZE = WIN_SIZE

        self.window = collections.deque(maxlen=WIN_SIZE)

    def clear(self):
        self.window.clear()
        self.mean = 0
        self.run_var = 0

    def is_full(self):
        return len(self.window) == self.WIN_SIZE

    def push(self, x):

        if len(self.window) == self.WIN_SIZE:
            # Adjusting variance
            x_removed = self.window.popleft()
            self.window.append(x)
            old_m = self.mean
            self.mean += (x - x_removed) / self.WIN_SIZE
            self.run_var += (x + x_removed - old_m - self.mean) * (x - x_removed)
        else:
            # Calculating first variance
            self.window.append(x)
            delta = x - self.mean
            self.mean += delta / len(self.window)
            self.run_var += delta * (x - self.mean)

    def get_mean(self):
        return self.mean if len(self.window) else 0.0

    def get_var(self):
        return self.run_var / len(self.window) if len(self.window) > 1 else 0.0

    def get_std(self):
        return math.sqrt(self.get_var())

    def get_all(self):
        return list(self.window)

    def __str__(self):
        return "Current window values: {}".format(list(self.window))

def to_price_paths(logreturns):
    shape = list(logreturns.shape)
    shape[-1] += 1
    
    exp_logreturns = np.exp(logreturns)
    price_paths = np.ones(shape)
    cumulative_products = np.cumprod(exp_logreturns, axis=-1)
    
    price_paths[:,:,1:] = cumulative_products
    return price_paths

class GBM_Simulator:
    def __init__(self, dataset, nsamples=10):
        self.dataset = dataset
        self.ivol = dataset.df.ivol[1:] / np.sqrt(252) / 100
        self.rates = dataset.df.rates / np.sqrt(252)
        self.window = dataset.output_size
        self.train_size = len(dataset.Y_train)
        self.test_size = len(dataset.Y_test)
        self.nsamples = nsamples
    
    def _empirical_martingale_correction(self, paths, r):
        """
        paths: 2d array/list of dimensions nsamples * nsteps
        r:     risk-free rate
        """
        old_paths = paths.copy()
        for j in range(1, paths.shape[1]):
            Zj = paths[:, j-1] * paths[:, j] / old_paths[:, j-1]
            Z0 = 1/len(paths) * np.exp(-r/252*j) * Zj.sum()
            paths[:, j] = paths[0][0] * Zj / Z0
        return paths
    
    def _simulate(self, mu=0, var=0.1, s0=1, nsamples=10, nsteps=100, EMS=True):
        """
        Returns sample_paths, an np.array() of dimensions (nsamples * nsteps)
        """
        # For each sample
        init_path = np.zeros(nsteps + 1)
        init_path[0] = s0
        sample_paths = np.array([init_path for _ in range(nsamples)]) # nsamples * (nsteps+1)
        
        # Adjust drift terms
        nu = mu - var/2
        std = np.sqrt(var)
        # For each sample
        for nsample in range(0, nsamples, 2):
            for nstep in range(nsteps):
                # Generate correlated random variables
                z = np.random.randn()
                # Get normal & antithetic steps
                step = nu + z*std
                antithetic_step = nu - z*std
                # Update the paths
                sample_paths[nsample, nstep+1] = sample_paths[nsample, nstep] * np.exp(step)
                if nsample+1<nsamples: sample_paths[nsample+1, nstep+1] = sample_paths[nsample+1, nstep] * np.exp(antithetic_step)
        # Empirical martingale correction
        if EMS: sample_paths = self._empirical_martingale_correction(sample_paths, r=mu)
        return sample_paths

    def _calibrate(self, logreturns):
        # Get drift
        mu = logreturns.mean()
        # Get diffusion
        var = logreturns.var()
        return mu, var
    
    def run(self, risk_neutral=True, EMS=True):
        simulations = np.zeros((self.test_size, self.nsamples, self.window+1))
        # For each window
        for i in tqdm(range(self.test_size-1), position=0, leave=True):
            var = self.ivol.iloc[i] ** 2
            mu = self.rates.iloc[i]
            # For each sample
            simulations[i,:,:] = self._simulate(mu, var, nsamples=self.nsamples, nsteps=self.window, EMS=EMS)
        return simulations
    
    def get_sims(self, date, n=100):
        ivol = self.ivol.loc[date]
        return self._simulate(mu=0, var=ivol**2, nsamples=n, nsteps=self.window, EMS=True)

# Define a function to compute the KL divergence
def kl_divergence(pdf1, pdf2):
    # Sample
    samples = pdf1.resample(size=10_000)
    pdf1_values = pdf1(samples)
    pdf2_values = pdf2(samples)
    # Compute the log ratio of the PDF values, avoid division by zero
    with np.errstate(divide='ignore'):
        log_ratios = np.log(pdf1_values/pdf2_values)
    # Compute the Monte Carlo estimate of the KL divergence
    log_ratios = log_ratios[~(np.isnan(log_ratios)|np.isinf(log_ratios))]
    return np.mean(log_ratios)

def js_divergence(data1, data2, verbose=True, bw_method=None, **kwargs):
    data1 = data1.reshape((-1,data1.shape[0]))
    data2 = data2.reshape((-1,data2.shape[0]))
    data1 = data1[:, ~np.any(np.isnan(data1)|np.isinf(data1), axis=0)]
    data2 = data2[:, ~np.any(np.isnan(data2)|np.isinf(data2), axis=0)]

    min_len = min(data1.shape[1], data2.shape[1])
    np.random.shuffle(data1)
    np.random.shuffle(data2)
    data1 = data1[:, :min_len]
    data2 = data2[:, :min_len]
    pdf1 = gaussian_kde(data1, bw_method=bw_method)
    pdf2 = gaussian_kde(data2, bw_method=bw_method)

    kl1 = kl_divergence(pdf1, pdf2, **kwargs)
    if verbose: print(f"KL(data1||data2) = {kl1}")
    kl2 = kl_divergence(pdf2, pdf1, **kwargs)
    if verbose: print(f"KL(data2||data1) = {kl2}")
    return (kl1+kl2)/2

def fid(data1, data2):
    data1 = data1.reshape((data1.shape[0], -1))[:,1:]
    data2 = data2.reshape((data2.shape[0], -1))[:,1:]
    # calculate mean and covariance statistics
    mu1, sigma1 = data1.mean(axis=0), np.cov(data1, rowvar=False)
    mu2, sigma2 = data1.mean(axis=0), np.cov(data2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = 0.5 * (sqrtm(sigma1.dot(sigma2)) + sqrtm(sigma2.dot(sigma1)))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid

def to_pkl(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
def from_pkl(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj