# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import logging
import operator
import os
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from imageio import imsave
from utils.utils import make_grid, save_image
from tqdm import tqdm
import cv2

import os
import tensorflow as tf

logger = logging.getLogger(__name__)
cuda_available = torch.cuda.is_available()
device = torch.device(f"cuda" if cuda_available else "cpu")

def cur_stages(iter, args):
        """
        Return current stage.
        :param epoch: current epoch.
        :return: current stage
        """
        idx = 0
        for i in range(len(args.grow_steps)):
            if iter >= args.grow_steps[i]:
                idx = i+1
        return idx

def compute_gradient_penalty(D, real_samples, conditions, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.shape[0], 1, 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, conditions)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty

def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader,
          epoch, writer_dict, fixed_z, schedulers=None):
    writer = writer_dict['writer']
    all_train_conditions = train_loader.dataset.X_train
    all_train_imgs = train_loader.dataset.Y_train
    # train_size = all_train_conditions.shape[0]
    # all_unique_train_conditions = np.unique(all_train_conditions.reshape(train_size,-1), axis=0).reshape(train_size,1,1,-1)
    gen_step = 0

    # train mode
    gen_net.train()
    dis_net.train()
    
    dis_optimizer.zero_grad()
    gen_optimizer.zero_grad()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    for iter_idx, (batch_train_conditions, batch_train_imgs) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']
        batch_size = batch_train_conditions.shape[0]
        total_dis_points = batch_size*args.dis_sample_size
        total_gen_points = batch_size*args.gen_sample_size

        ## randomly draw batch_size y's from unique_conditions
        real_conditions = batch_train_conditions.numpy()
        ## add Gaussian noise; we estimate image distribution conditional on these labels
        epsilons = np.concatenate([np.random.normal(0, sigma, (batch_size,1,1,1)) for sigma in args.kernel_sigma], axis=-1)
        # ## uniform noise
        # lb = np.reshape(target_conditions[j], (-1)) - args.kernel_sigma
        # ub = np.reshape(target_conditions[j], (-1)) + args.kernel_sigma
        # np.concatenate([np.random.uniform(l,u, (1,1,1)) for l,u in zip(lb,ub)], axis=-1)
        target_conditions = real_conditions + epsilons

        ## find index of real images with labels in the vicinity of batch_target_labels
        ## generate labels for fake image generation; these labels are also in the vicinity of batch_target_labels
        real_idx = np.zeros((total_dis_points), dtype=int) #index of images in the data; the labels of these images are in the vicinity
        for j in range(batch_size):
            sample_start = j*args.dis_sample_size
            sample_end = (j+1)*args.dis_sample_size

            dist_to_target = np.reshape(((all_train_conditions-target_conditions[j])**2).sum(axis=-1), (-1,))
            idx_in_vicinity = np.argsort(dist_to_target)[:args.dis_sample_size]

            # ## index for real images
            # is_in_vicinity = np.reshape(((all_train_conditions-target_conditions[j])**2).sum(axis=-1), (-1,)) <= args.kappa
            # idx_in_vicinity = np.nonzero(is_in_vicinity)[0]

            # ## if the max gap between two consecutive ordered unique labels is large, it is possible that len(indx_real_in_vicinity)<1
            # while len(idx_in_vicinity)<args.dis_sample_size:
            #     # redefine batch_target_labels
            #     epsilons_j = np.concatenate([np.random.normal(0, sigma, (1,1,1)) for sigma in args.kernel_sigma], axis=-1)
            #     target_conditions[j] = real_conditions[j] + epsilons_j
          
            #     ## index for real images
            #     is_in_vicinity = np.reshape(((all_train_conditions-target_conditions[j])**2).sum(axis=-1), (-1,)) <= args.kappa
            #     idx_in_vicinity = np.nonzero(is_in_vicinity)[0]

            #     print("IN VICINITY", len(idx_in_vicinity), real_conditions[j])
            
            # select the real image
            real_idx[sample_start:sample_end] = idx_in_vicinity # np.random.choice(idx_in_vicinity, size=args.dis_sample_size)[0]

        ## draw the real image batch from the training set
        real_imgs = torch.from_numpy(all_train_imgs[real_idx]).to(device, dtype=torch.float)
        # real_conditions = torch.from_numpy(all_train_conditions[real_idx]).to(device, dtype=torch.float)
        target_conditions = torch.from_numpy(target_conditions).to(device, dtype=torch.float) # shape: (batch_size,1,1,num_conditions)
        dis_target_conditions = target_conditions.repeat_interleave(args.dis_sample_size, dim=0) # shape: (total_dis_points,1,1,num_conditions)
        gen_target_conditions = target_conditions.repeat_interleave(args.gen_sample_size, dim=0) # shape: (total_gen_points,1,1,num_conditions)

        # # Adversarial ground truths
        # real_imgs = imgs
        # if cuda_available: real_imgs = real_imgs.type(torch.cuda.FloatTensor).cuda(args.gpu, non_blocking=True)
        # else: real_imgs = real_imgs.type(torch.FloatTensor).to(device, non_blocking=True)

        # # Conditions
        # real_conditions = conditions
        # if cuda_available: real_conditions = torch.cuda.FloatTensor(real_conditions).cuda(args.gpu, non_blocking=True)
        # else: real_conditions = torch.FloatTensor(real_conditions).to(device, non_blocking=True)

        # Sample noise as generator input
        z = np.random.normal(0, 1, (total_dis_points, args.latent_dim))
        z = torch.FloatTensor(z).to(device, non_blocking=True)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        fake_imgs = gen_net(z, dis_target_conditions).detach()
        real_validity = dis_net(real_imgs, dis_target_conditions)
        fake_validity = dis_net(fake_imgs, dis_target_conditions)

        d_loss_components = {}
        # cal loss
        d_loss_components['real'] = -torch.mean(real_validity)
        d_loss_components['fake'] = torch.mean(fake_validity)
        d_loss_components['gp'] = compute_gradient_penalty(dis_net, real_imgs, dis_target_conditions, fake_imgs.detach(), args.phi) * 10 / (args.phi ** 2)
        # d_loss_components['eps'] = (torch.mean(real_validity) ** 2) * 1e-3 # Adds eps
        
        d_loss = 0
        for key,val in d_loss_components.items():
            d_loss_components[key] = val/float(args.accumulated_times)
            d_loss = d_loss + d_loss_components[key]
        d_loss.backward(retain_graph=True)
        
        if (iter_idx + 1) % args.accumulated_times == 0:
            torch.nn.utils.clip_grad_norm_(dis_net.parameters(), 5.)
            dis_optimizer.step()
            dis_optimizer.zero_grad()
            writer.add_scalar('d_loss', d_loss.item(), global_steps) if args.rank == 0 else 0

        # -----------------
        #  Train Generator
        # -----------------
        g_loss_components = {}
        if global_steps % (args.n_critic * args.accumulated_times) == 0:
            
            for accumulated_idx in range(args.g_accumulated_times):
                z = np.random.normal(0, 1, (total_gen_points, args.latent_dim))
                z = torch.FloatTensor(z).to(device)
                fake_imgs = gen_net(z, gen_target_conditions)
                ## sample subset of fake_imgs
                subset_idx = torch.cat([torch.randperm(args.gen_sample_size)[:args.dis_sample_size] + i*args.dis_sample_size for i in range(args.batch_size)])
                fake_validity = dis_net(fake_imgs[subset_idx], dis_target_conditions)
                real_validity = dis_net(real_imgs, dis_target_conditions).detach()
                
                g_loss_components['var'] = 0
                g_loss_components['m1'], g_loss_components['m2'], g_loss_components['m3'] = 0, 0, 0
                g_loss_components['m2x'] = 0

                g_loss_components['m1_val'] = 0
                g_loss_components['m2_val'] = 0
                g_loss_components['m3_val'] = 0

                for j in range(batch_size):
                    dis_sample_start = j*args.dis_sample_size
                    dis_sample_end = (j+1)*args.dis_sample_size
                    gen_sample_start = j*args.gen_sample_size
                    gen_sample_end = (j+1)*args.gen_sample_size
                    batch_real_imgs = real_imgs[dis_sample_start:dis_sample_end,0,0,1:]
                    batch_fake_imgs = fake_imgs[gen_sample_start:gen_sample_end,0,0,1:]

                    mu_real, mu_fake = torch.mean(batch_real_imgs, dim=0), torch.mean(batch_fake_imgs, dim=0)
                    std_real, std_fake = torch.std(batch_real_imgs, dim=0), torch.std(batch_fake_imgs, dim=0)
                    zscores_real, zscores_fake = (batch_real_imgs-mu_real)/std_real, (batch_fake_imgs-mu_fake)/std_fake
                    scaled_std_real, scaled_std_fake = std_real/mu_real, std_fake/mu_fake
                    n_real, n_fake = batch_real_imgs.size(0), batch_fake_imgs.size(0)
                    skew_real, skew_fake = torch.sum(torch.pow(zscores_real, 3.0), dim=0) * n_real/(n_real-1)/(n_real-2), torch.sum(torch.pow(zscores_fake, 3.0), dim=0) * n_fake/(n_fake-1)/(n_fake-2)
                    # skew_real, skew_fake = torch.mean(torch.pow(zscores_real, 3.0), dim=0), torch.mean(torch.pow(zscores_fake, 3.0), dim=0)

                    g_loss_components['var'] = g_loss_components['var'] + 1/(torch.mean(scaled_std_fake))
                    
                    g_loss_components['m1'] = g_loss_components['m1'] + torch.mean(torch.pow(mu_real-mu_fake, 2.0))
                    g_loss_components['m2'] = g_loss_components['m2'] + torch.mean(torch.pow(std_real-std_fake, 2.0))
                    g_loss_components['m3'] = g_loss_components['m3'] + torch.mean(torch.pow(skew_real-skew_fake, 2.0))

                    # Get only non-diagonal elements of corr matrix
                    corr_real, corr_fake = torch.corrcoef(batch_real_imgs.T), torch.corrcoef(batch_fake_imgs.T)
                    nsteps = batch_real_imgs.size(1)
                    assert nsteps == batch_fake_imgs.size(1), f"batch_real_imgs has {nsteps} timesteps, but batch_fake_imgs has {batch_fake_imgs.size(1)} timesteps."
                    # mask = 1 - torch.eye(nsteps, device=corr_real.device)
                    # g_loss_components['m2x'] = g_loss_components['m2x'] + torch.sum(torch.pow((corr_real-corr_fake)*mask, 2.0)) / (nsteps**2 - nsteps)
                    g_loss_components['m2x'] = g_loss_components['m2x'] + fid(batch_real_imgs, batch_fake_imgs)


                    batch_real_val = real_validity[dis_sample_start:dis_sample_end, :]
                    batch_fake_val = fake_validity[dis_sample_start:dis_sample_end, :]
                    mu_real_val, mu_fake_val = torch.mean(batch_real_val), torch.mean(batch_fake_val)
                    std_real_val, std_fake_val = torch.std(batch_real_val), torch.std(batch_fake_val)
                    zscores_real_val, zscores_fake_val = (batch_real_val-mu_real_val)/std_real_val, (batch_fake_val-mu_fake_val)/std_fake_val
                    # scaled_std_real_val, scaled_std_fake_val = std_real_val/mu_real_val, std_fake_val/mu_fake_val
                    skew_real_val, skew_fake_val = torch.sum(torch.pow(zscores_real_val, 3.0)) * n_real/(n_real-1)/(n_real-2), torch.sum(torch.pow(zscores_fake_val, 3.0)) * n_real/(n_real-1)/(n_real-2)
                    g_loss_components['m1_val'] = g_loss_components['m1_val'] + torch.pow(mu_real_val-mu_fake_val, 2.0)
                    g_loss_components['m2_val'] = g_loss_components['m2_val'] + torch.pow(std_real_val-std_fake_val, 2.0)
                    g_loss_components['m3_val'] = g_loss_components['m3_val'] + torch.pow(skew_real_val-skew_fake_val, 2.0)

                if args.var_term_weight==0: del g_loss_components['var']
                else: g_loss_components['var'] = g_loss_components['var'] * args.var_term_weight

                g_loss_components['m2x'] = g_loss_components['m2x'] * args.m2x_term_weight

                g_loss = 0
                for key,val in g_loss_components.items():
                    # assert val>0, f"{key} has negative value {val}"
                    g_loss_components[key] = val / batch_size / float(args.accumulated_times)
                    g_loss = g_loss + g_loss_components[key]
                    
                g_loss.backward()
                
            
            torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.)
            gen_optimizer.step()
            gen_optimizer.zero_grad()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            ema_nimg = args.ema_kimg * 1000
            cur_nimg = args.dis_batch_size * args.world_size * global_steps
            if args.ema_warmup != 0:
                ema_nimg = min(ema_nimg, cur_nimg * args.ema_warmup)
                ema_beta = 0.5 ** (float(args.dis_batch_size * args.world_size) / max(ema_nimg, 1e-8))
            else:
                ema_beta = args.ema
                
            # moving average weight\
            # Iterate directly over the tensor
            for p, avg_p in zip(gen_net.parameters(), (avg_p for _, avg_p in gen_avg_param.items()) if isinstance(gen_avg_param, dict) else gen_avg_param):
                cpu_p = deepcopy(p)
                avg_p.mul_(ema_beta).add_(1. - ema_beta, cpu_p.to(avg_p.device).data)
                del cpu_p
            
            writer.add_scalar('g_loss', g_loss.item(), global_steps) if args.rank == 0 else 0
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0 and args.rank == 0:
            sample_imgs = torch.cat((fake_imgs[:16], real_imgs[:16]), dim=0)
#             scale_factor = args.img_size // int(sample_imgs.size(3))
#             sample_imgs = torch.nn.functional.interpolate(sample_imgs, scale_factor=2)
#             img_grid = make_grid(sample_imgs, nrow=4, normalize=True, scale_each=True)
#             save_image(sample_imgs, f'sampled_images_{args.exp_name}.jpg', nrow=4, normalize=True, scale_each=True)
            # writer.add_image(f'sampled_images_{args.exp_name}', img_grid, global_steps)
            d_loss_components_str = ", ".join([f"{key}: {val.item()}" for key,val in d_loss_components.items()])
            g_loss_components_str = ", ".join([f"{key}: {val.item()}" for key,val in g_loss_components.items()])
            fake_vals = fake_imgs.detach()
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f (%s)] [G loss: %f (%s)] [ema: %f] [gen~N(%f,%f)]" %
                (epoch+1, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), d_loss_components_str, g_loss.item(), g_loss_components_str, ema_beta,
                 fake_vals.mean(), fake_vals.var())
            )
            del fake_imgs
            del real_imgs
            del fake_validity
            del real_validity
            del g_loss
            del d_loss

        writer_dict['train_global_steps'] = global_steps + 1


def save_samples(args, fixed_z, epoch, gen_net: nn.Module, writer_dict, clean_dir=True):
    # eval mode
    gen_net.eval()
    with torch.no_grad():
        # generate images
        batch_size = fixed_z.size(0)
        sample_imgs = []
        for i in range(fixed_z.size(0)):
            sample_img = gen_net(fixed_z[i:(i+1)], epoch)
            sample_imgs.append(sample_img)
        sample_imgs = torch.cat(sample_imgs, dim=0)
        os.makedirs(f"./samples/{args.exp_name}", exist_ok=True)
        save_image(sample_imgs, f'./samples/{args.exp_name}/sampled_images_{epoch}.png', nrow=10, normalize=True, scale_each=True)
    return 0

class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):
        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr

def load_params(model, new_param, args, mode="gpu"):
    if mode == "cpu":
        for p, new_p in zip(model.parameters(), new_param):
            cpu_p = deepcopy(new_p)
            # p.data.copy_(cpu_p.cuda().to(f"cuda:{args.gpu}"))
            p.data.copy_(cpu_p.cuda().to("cpu"))
            del cpu_p
    
    else:
        for p, new_p in zip(model.parameters(), new_param.values() if isinstance(new_param, dict) else new_param):
            p.data.copy_(new_p)


def copy_params(model, mode='cpu'):
    if type(model) is dict or type(model) is OrderedDict:
        new_state_dict = OrderedDict()
        for param_name, param_value in model.items():
            # Your logic to copy and potentially move the parameter to GPU
            # For example, if mode is 'gpu', you might do something like:
            new_state_dict[param_name] = param_value.to(device)
        return new_state_dict
    else:
        if mode == 'gpu':
            flatten = []
            for p in model.parameters():
                cpu_p = deepcopy(p).cpu()
                flatten.append(cpu_p.data)
        else:
            flatten = deepcopy(list(p.data for p in model.parameters()))
        return flatten

def pearson_corr(y1, y2):
    """Compute Pearson correlation coefficient between two tensors."""
    mean_x = torch.mean(y1)
    mean_y = torch.mean(y2)
    
    m1 = y1.sub(mean_x)
    m2 = y2.sub(mean_y)
    
    num = torch.sum(m1 * m2)
    den = torch.sqrt(torch.sum(m1 ** 2) * torch.sum(m2 ** 2))
    return num/den

def to_price_paths(logreturns):
    shape = list(logreturns.shape)
    shape[-1] += 1

    exp_logreturns = torch.exp(logreturns)
    price_paths = torch.ones(shape, dtype=torch.float, device=device)
    cumulative_products = torch.cumprod(exp_logreturns, dim=-1)
    
    price_paths[:,:,:,1:] = cumulative_products
    return price_paths

def fid(data1, data2):
    # calculate mean and covariance statistics
    # mu1, mu2 = torch.mean(data1, dim=0), torch.mean(data2, dim=0)
    sigma1, sigma2 = torch.cov(data1.T), torch.cov(data2.T)
    # if var_multiple!=1: sigma2 = sigma2 * (torch.eye(sigma2.size(0)).to(device) * (1/var_multiple-1) + torch.ones_like(sigma2).to(device))
    # calculate sqrt of product between cov
    eigvals = torch.linalg.eigvals(sigma1@sigma2).sqrt().real
    # calculate score
    # means_l2 = sum((mu1 - mu2)**2.0)
    covar_sum = torch.trace(sigma1) + torch.trace(sigma2) - 2*eigvals.sum()

    # fid_score = means_l2 + covar_sum
    return covar_sum # fid_score

def torch_sqrtm(matrix):
    """Compute the square-root of a positive semi-definite matrix using eigenvalue decomposition."""
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    # Compute the square root of the eigenvalues
    sqrt_eigenvalues = torch.clamp(eigenvalues, min=1e-20).sqrt()
    # Reconstruct the square root matrix
    covmean = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
    return covmean

def adapt_state_dict_for_loading(state_dict, model):
    new_state_dict = OrderedDict()
    is_data_parallel_model = isinstance(model, torch.nn.DataParallel)

    for key, value in state_dict.items():
        if is_data_parallel_model and not key.startswith('module.'):
            # Add 'module.' prefix
            new_key = 'module.' + key
        elif not is_data_parallel_model and key.startswith('module.'):
            # Remove 'module.' prefix
            new_key = key[len('module.'):]
        else:
            # No change needed
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict