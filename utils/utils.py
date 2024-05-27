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
import json
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.stats import norm
import re
import QuantLib as ql
from typing import Union, Optional, List, Tuple, Text, BinaryIO
import pathlib
import torch
import math
import warnings
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
from scipy.stats import ncx2

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

def to_json(filepath, obj):
    with open(filepath, 'w') as f:
        json.dump(obj, f)
def from_json(filepath):
    with open(filepath) as f:
        obj = json.load(f)
    return obj

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

class Simulator(ABC):
    def __init__(self, dataset, nsamples=10):
        self.dataset = dataset
        self.window = dataset.output_size
        self.train_size = len(dataset.Y_train)
        self.test_size = len(dataset.option_prices.loc[dataset.test_start_date:].index)
        self.nsamples = nsamples
        self.ivol_surface = dataset.df[list(dataset.ivol_surface.columns)] / np.sqrt(252) / 100
        self.ivol = self.ivol_surface['100%60d']
        self.option_prices = dataset.option_prices
        self.rates = np.log(1+dataset.df.rates[self.train_size:]/100) / 252
    
    def _empirical_martingale_correction(self, paths, r):
        """m
        paths: 2d array/list of dimensions nsamples * nsteps
        r:     risk-free rate
        """
        old_paths = paths.copy()
        for j in range(1, paths.shape[1]):
            Zj = paths[:, j-1] * paths[:, j] / old_paths[:, j-1]
            Z0 = 1/len(paths) * np.exp(-r*j) * Zj.sum()
            paths[:, j] = paths[0][0] * Zj / Z0
        return paths
    
    def _calibrate(self, logreturns):
        # Get drift
        mu = logreturns.mean()
        # Get diffusion
        var = logreturns.var()
        return mu, var
    
    def BS_theoretical_call_price(self, S, K, T, mu, sigma, r):
        """Calculate the Black-Scholes price for a European call option."""
        d1 = (np.log(S / K) + (mu + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price

    def BS_theoretical_put_price(self, S, K, T, mu, sigma, r):
        """Calculate the Black-Scholes price for a European put option."""
        d1 = (np.log(S / K) + (mu + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price
    
    @abstractmethod
    def _simulate(self):
        pass


class GBM_Simulator(Simulator):
    def __init__(self, dataset, nsamples=10):
        super().__init__(dataset=dataset, nsamples=nsamples)

    def _simulate(self, mu, sigma, s0=1, nsamples=10, nsteps=100, EMS=True):
        """
        Returns paths, an np.array() of dimensions (nsamples * nsteps)
        """
        paths = np.zeros((nsamples, nsteps + 1)) # nsamples * (nsteps+1)
        paths[:, 0] = s0
        
        # Adjust drift terms
        nu = mu - (sigma**2)/2
        # For each sample
        for i in range(0, nsamples, 2):
            add_antithetic = i+1<nsamples
            for t in range(nsteps):
                # Generate rv
                z = np.random.normal()
                # Get incremental step
                step = nu + z*sigma
                # Update the paths
                paths[i, t+1] = paths[i, t] * np.exp(step)

                if add_antithetic:
                    # Get antithetic variates
                    antithetic_step = nu - z*sigma
                    paths[i+1, t+1] = paths[i+1, t] * np.exp(antithetic_step)
        # Empirical martingale correction
        if EMS: paths = self._empirical_martingale_correction(paths, r=mu)
        return paths
    
    def run(self):
        simulations = np.zeros((self.test_size, self.nsamples, self.window+1))
        # For each window
        for i in range(self.test_size):
            simulations[i,:,:] = self.get_sims(i, n=self.nsamples, mode='index')
        return simulations
    
    def get_sims(self, i, n=100, mode='date'):
        if mode=='date':
            mu = self.rates.loc[i]
            sigma = self.ivol.loc[i]
        else:
            mu = self.rates.iloc[i]
            sigma = self.ivol.iloc[i]
        return self._simulate(mu=mu, sigma=sigma, nsamples=n, nsteps=self.window, EMS=True)
    

class CEV_Simulator(Simulator):
    def __init__(self, dataset, nsamples=10):
        super().__init__(dataset=dataset, nsamples=nsamples)

    def _simulate(self, params, s0=1, nsamples=10, nsteps=100, EMS=True):
        """
        Returns paths, an np.array() of dimensions (nsamples * nsteps)
        """
        mu, sigma, gamma = params
        paths = np.zeros((nsamples, nsteps + 1)) # nsamples * (nsteps+1)
        paths[:, 0] = s0
        
        # For each sample
        for i in range(0, nsamples, 2):
            add_antithetic = i+1<nsamples
            for t in range(nsteps):
                # Generate rv
                z = np.random.normal()
                # Get incremental step
                drift = mu - 0.5 * sigma**2 * np.exp((2*gamma - 2) * paths[i, t])
                if np.isinf(drift):
                    if drift<0: drift = -1
                    else: drift = 1
                diffusion = min(sigma * np.exp((gamma - 1) * paths[i, t]), 1)
                step = drift + z*diffusion
                # Update the paths
                paths[i, t+1] = paths[i, t] * np.exp(step)

                if add_antithetic: 
                    # Get antithetic variates
                    antithetic_step = drift - z*diffusion
                    paths[i+1, t+1] = paths[i+1, t] * np.exp(antithetic_step)
        # Empirical martingale correction
        if EMS: paths = self._empirical_martingale_correction(paths, r=mu)
        return paths
    
    def _price(self, S, K, mu, sigma, gamma, r, T, option_type='call'):
        """
        Calculate the theoretical CEV implied price for an option given parameters.
        
        Parameters:
        S (float): Current stock price
        K (float): Strike price of the option
        mu (float): Drift term in the CEV model
        sigma (float): Volatility coefficient in the CEV model
        gamma (float): Elasticity parameter
        r (float): Risk-free rate
        T (float): Time to expiration of the option
        option_type (str): 'call' for call option, 'put' for put option
        """

        collapse_to_BS = gamma==1
        if option_type=='call':
            # theoretical_price = self.BS_theoretical_call_price(S, K, T, mu, sigma, r)
            if collapse_to_BS:
                theoretical_price = self.BS_theoretical_call_price(S, K, T, mu, sigma, r)
            else:
                a = (K**(2 * (1-gamma))) / ((1-gamma)**2 * sigma**2 * T)
                b = 1 / (1 - gamma)
                c = ((S*np.exp(mu*T)) ** (2*(1-gamma))) / ((1-gamma)**2 * sigma**2 * T)
                if gamma < 1:
                    theoretical_price = S*(1 - ncx2.cdf(a,b+2,c)) - K*np.exp(-r*T)*ncx2.cdf(c,b,a)
                else:
                    theoretical_price = S*(1 - ncx2.cdf(c,-b,a)) - K*np.exp(-r*T)*ncx2.cdf(a,2-b,a)
        elif option_type=='put':
            # theoretical_price = self.BS_theoretical_put_price(S, K, T, mu, sigma, r)
            if collapse_to_BS:
                theoretical_price = self.BS_theoretical_put_price(S, K, T, mu, sigma, r)
            else:
                a = (K**(2 * (1-gamma))) / ((1-gamma)**2 * sigma**2 * T)
                b = 1 / (1 - gamma)
                c = ((S*np.exp(mu*T)) ** (2*(1-gamma))) / ((1-gamma)**2 * sigma**2 * T)
                if gamma < 1:
                    theoretical_price = K*np.exp(-r*T)*(1 - ncx2.cdf(c,b,a)) - S*ncx2.cdf(a,b+2,c)
                else:
                    theoretical_price = K*np.exp(-r*T)*(1 - ncx2.cdf(a,2-b,c)) - S*ncx2.cdf(c,-b,a)
        else:
            raise NotImplementedError
        
        return theoretical_price
    
    def _objective_func(self, params, market_prices, moneyness_levels, maturities, option_types, r, S0=1):
        """
        Objective function to minimize the sum of squared differences between market and model average prices.
        """
        max_T = int(max(maturities) * 252/365)
        paths = self._simulate(params, nsamples=100, nsteps=max_T, EMS=True)
    
        errors = []
        for i, (market_price, M, T, option_type) in enumerate(zip(market_prices, moneyness_levels, maturities, option_types)):
            K = S0*M
            T = int(T*252/365)
            if option_type=='call':
                model_price = (np.maximum(paths[:,T] - K, 0)).mean() * np.exp(-r*T)
            elif option_type=='put':
                model_price = (np.maximum(K - paths[:,T], 0)).mean() * np.exp(-r*T)
            # model_price = self._price(S0, K, mu, sigma, gamma, r, T*252/365, option_type)
            # print(model_price, market_price)
            error = (model_price-market_price)**2
            errors.append(error)
        return np.sum(errors)

    def _calibrate(self, i, mode='index'):
        if mode=='index':
            option_eod = self.option_prices.iloc[i]
            ivol_surface = self.ivol_surface.iloc[i]
            r = self.rates.iloc[i]
        else:
            option_eod = self.option_prices.loc[i]
            ivol_surface = self.ivol_surface.loc[i]
            r = self.rates.loc[i]

        moneyness_level = option_eod['moneyness_level']
        maturity = option_eod['maturity']
        S0 = option_eod['underlying_price']
        args=([option_eod['call_price'], option_eod['put_price']], [moneyness_level, moneyness_level], [maturity, maturity], ['call', 'put'], r, S0)
        initial_params = [r, ivol_surface.mean(), 1] # mu, sigma, gamma
        est_gamma = minimize(self._objective_func, initial_params, args=args, bounds=[(r,r), (1e-3,1e-2), (0,2)])
        return est_gamma.x
    
    def run(self):
        simulations = np.zeros((self.test_size, self.nsamples, self.window+1))
        # For each window
        for i in range(self.test_size):
            simulations[i,:,:] = self.get_sims(i, n=self.nsamples, mode='index')
        return simulations
    
    def get_sims(self, i, n=100, mode='date'):
        params = self._calibrate(i, mode=mode)
        # print(params)
        # print()
        return self._simulate(params, nsamples=n, nsteps=self.window, EMS=True)


class Heston_Simulator(Simulator):
    def __init__(self, dataset, nsamples=10):
        super().__init__(dataset=dataset, nsamples=nsamples)

    def _simulate(self, params, s0=1, nsamples=10, nsteps=100, EMS=True):
        """
        Returns paths, an np.array() of dimensions (nsamples * nsteps)
        """
        mu, kappa, theta, sigma, rho, v0 = params
        # For each sample
        paths = np.zeros((nsamples, nsteps + 1)) # nsamples * (nsteps+1)
        paths[:, 0] = s0
        variances = np.zeros((nsamples, nsteps + 1)) # nsamples * (nsteps+1)
        variances[:, 0] = v0

        # For each sample
        for i in range(0, nsamples, 2):
            add_anithetic = i+1<nsamples
            for t in range(nsteps):
                # Generate correlated Brownian motions
                z1 = np.random.normal()
                z2 = np.random.normal()
                dW_S = z1
                dW_v = rho * dW_S + np.sqrt(1 - rho**2) * z2
                v_prev = variances[i, t]
                variances[i, t+1] = np.maximum(v_prev + kappa * (theta - v_prev) + sigma * np.sqrt(np.maximum(v_prev, 0)) * dW_v, 0) # Ensure variance stays positive
                paths[i, t+1] = paths[i, t] * np.exp((mu - 0.5 * v_prev) + np.sqrt(np.maximum(v_prev, 0)) * dW_S)

                if add_anithetic:
                    # For the antithetic variate
                    antithetic_dW_S = -z1
                    antithetic_dW_v = rho * antithetic_dW_S + np.sqrt(1 - rho**2) * -z2
                    v_prev = variances[i+1, t]
                    variances[i+1, t+1] = np.maximum(v_prev + kappa * (theta - v_prev) + sigma * np.sqrt(np.maximum(v_prev, 0)) * antithetic_dW_v, 0)
                    paths[i+1, t+1] = paths[i+1, t] * np.exp((mu - 0.5 * variances[i + 1, t+1]) + np.sqrt(np.maximum(variances[i+1, t+1], 0)) * antithetic_dW_S)
                
        # Empirical martingale correction
        if EMS: paths = self._empirical_martingale_correction(paths, r=mu)
        return paths
    
    def _init_engine(self, params, r, cur_date, S0=1):
        mu, kappa, theta, sigma, rho, v0 = params
        # Check that params satisfy Feller condition
        assert 2*kappa*theta > sigma**2, f"Your kappa ({kappa}), theta ({theta}) and sigma ({sigma}) doesn't satisfy the Feller condition 2*kappa*theta > sigma**2.\nThis means your stochastic variance vt might reach negative values and cause problems to the Heston engine."
        cur_date = ql.Date(cur_date.day, cur_date.month, cur_date.year)
        ql.Settings.instance().evaluationDate = cur_date

        spot_price = ql.SimpleQuote(S0)
        rf_rate = ql.SimpleQuote(r)
        yield_ts = ql.FlatForward(cur_date, ql.QuoteHandle(rf_rate), ql.Actual365Fixed())
        risk_free_curve = ql.YieldTermStructureHandle(yield_ts)

        # Heston model setup
        heston_process = ql.HestonProcess(risk_free_curve, ql.YieldTermStructureHandle(ql.FlatForward(cur_date, 0, ql.Actual365Fixed())),
                                        ql.QuoteHandle(spot_price), v0, kappa, theta, sigma, rho)
        heston_model = ql.HestonModel(heston_process)
        engine = ql.AnalyticHestonEngine(heston_model)
        return engine
    
    def _price(self, S, K, T, cur_date, option_type='call'):
        """
        Calculate the theoretical Heston implied price for an option given parameters.
        
        Parameters:
        S (float): Current stock price
        K (float): Strike price of the option
        T (float): Time to expiration of the option
        cur_date
        option_type (str): 'call' for call option, 'put' for put option
        """
        cur_date = ql.Date(cur_date.day, cur_date.month, cur_date.year)
        expiry = cur_date + ql.Period(int(T), ql.Days)
        if option_type=='call':
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, float(K))
        elif option_type=='put':
            payoff = ql.PlainVanillaPayoff(ql.Option.Put, float(K))
        else:
            raise NotImplementedError
        exercise = ql.EuropeanExercise(expiry)
        european_option = ql.VanillaOption(payoff, exercise)
        european_option.setPricingEngine(self.engine)
        theoretical_price = european_option.NPV()
        return theoretical_price
    
    def _objective_func(self, params, market_prices, moneyness_levels, maturities, option_types, r, cur_date, S0=1):
        """
        Objective function to minimize the sum of squared differences between market and model average prices.
        """
        # print(params)
        # Check that params satisfy Feller condition
        mu, kappa, theta, sigma, rho, v0 = params
        LHS = 2*kappa*theta
        RHS = sigma**2
        if LHS <= RHS:
            errors_sum = (1000**2) * 2 * (RHS - LHS)/LHS
        else:
            self.engine = self._init_engine(params, r=r, cur_date=cur_date, S0=S0)
            errors = []
            for i, (market_price, M, T, option_type) in enumerate(zip(market_prices, moneyness_levels, maturities, option_types)):
                K = S0*M
                model_price = self._price(S0, K, T, cur_date, option_type)
                # print(model_price, market_price)
                error = (model_price-market_price)**2
                errors.append(error)
            errors_sum = np.sum(errors)
        # print(errors_sum)
        return errors_sum
    
    def _calibrate(self, i, mode='index'):
        if mode=='index':
            cur_date = self.option_prices.index[i]
            option_eod = self.option_prices.iloc[i]
            ivol_surface = self.ivol_surface.iloc[i]
            r = self.rates.iloc[i]
        else:
            cur_date = i
            option_eod = self.option_prices.loc[i]
            ivol_surface = self.ivol_surface.loc[i]
            r = self.rates.loc[i]

        moneyness_level = option_eod['moneyness_level']
        maturity = option_eod['maturity']
        S0 = option_eod['underlying_price']
        args=([option_eod['call_price'], option_eod['put_price']], [moneyness_level, moneyness_level], [maturity, maturity], ['call', 'put'], r, cur_date, S0)
        
        initial_v0 = ivol_surface.mean()**2
        initial_params = [r, 1e-5, initial_v0, 1e-5, 0, initial_v0] # mu, kappa, theta, sigma, rho, v0
        est_gamma = minimize(self._objective_func, initial_params, args=args, bounds=[(-1e-2,1e-2), (1e-10,0.3), (1e-10,1e-3), (1e-30,1e-3), (-1,1), (1e-30,0.5e-3)])
        return est_gamma.x
    
    def run(self):
        simulations = np.zeros((self.test_size, self.nsamples, self.window+1))
        # For each window
        for i in range(self.test_size):
            simulations[i,:,:] = self.get_sims(i, n=self.nsamples, mode='index')
        return simulations
    
    def get_sims(self, i, n=100, mode='date'):
        params = self._calibrate(i, mode=mode)
        # print(params)
        # print()
        return self._simulate(params, nsamples=n, nsteps=self.window, EMS=True)


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