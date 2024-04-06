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

# from utils.fid_score import calculate_fid_given_paths
from utils.torch_fid_score import get_fid
import os
import tensorflow as tf

logger = logging.getLogger(__name__)
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

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
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
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

        ## randomly draw batch_size y's from unique_conditions
        target_conditions_raw = batch_train_conditions.numpy() # unique_conditions[np.random.choice(range(len(unique_conditions)), size=batch_size, replace=True),:,:,:]
        ## add Gaussian noise; we estimate image distribution conditional on these labels
        epsilons = np.concatenate([np.random.normal(0, sigma, (batch_size,1,1,1)) for sigma in args.kernel_sigma], axis=-1)
        target_conditions = target_conditions_raw + epsilons

        ## find index of real images with labels in the vicinity of batch_target_labels
        ## generate labels for fake image generation; these labels are also in the vicinity of batch_target_labels
        real_idx = np.zeros(batch_size, dtype=int) #index of images in the datata; the labels of these images are in the vicinity
        fake_conditions = np.zeros_like(target_conditions_raw)

        for j in range(batch_size):
            ## index for real images
            is_in_vicinity = np.reshape(((all_train_conditions-target_conditions[j])**2).sum(axis=-1), (-1,)) <= args.kappa
            idx_real_in_vicinity = np.nonzero(is_in_vicinity)[0]

            ## if the max gap between two consecutive ordered unique labels is large, it is possible that len(indx_real_in_vicinity)<1
            while len(idx_real_in_vicinity)<1:
                # redefine batch_target_labels
                epsilons_j = np.concatenate([np.random.normal(0, sigma, (1,1,1)) for sigma in args.kernel_sigma], axis=-1)
                target_conditions[j] = target_conditions_raw[j] + epsilons_j
          
                ## index for real images
                is_in_vicinity = np.reshape(((all_train_conditions-target_conditions[j])**2).sum(axis=-1), (-1,)) <= args.kappa
                idx_real_in_vicinity = np.nonzero(is_in_vicinity)[0]
            
            # select the real image
            real_idx[j] = np.random.choice(idx_real_in_vicinity, size=1)[0]
            # then from the real image conditions, create the fake image conditions
            ## labels for fake images generation
            lb = np.reshape(target_conditions[j], (-1)) - args.kernel_sigma
            ub = np.reshape(target_conditions[j], (-1)) + args.kernel_sigma
            fake_conditions[j] = np.concatenate([np.random.uniform(l,u, (1,1,1)) for l,u in zip(lb,ub)], axis=-1)

        ## draw the real image batch from the training set
        real_imgs = torch.from_numpy(all_train_imgs[real_idx]).to(device, dtype=torch.float)
        real_conditions = torch.from_numpy(all_train_conditions[real_idx]).to(device, dtype=torch.float)
        target_conditions = torch.from_numpy(target_conditions).type(torch.float).to(device)
        target_conditions_reshaped = target_conditions.reshape(target_conditions.shape[0], -1)
        fake_conditions = torch.from_numpy(fake_conditions).type(torch.float).to(device)
        fake_conditions_reshaped = fake_conditions.reshape(fake_conditions.shape[0], -1)
        
        
        # # Adversarial ground truths
        # real_imgs = imgs
        # if cuda_available: real_imgs = real_imgs.type(torch.cuda.FloatTensor).cuda(args.gpu, non_blocking=True)
        # else: real_imgs = real_imgs.type(torch.FloatTensor).to(device, non_blocking=True)

        # # Conditions
        # real_conditions = conditions
        # if cuda_available: real_conditions = torch.cuda.FloatTensor(real_conditions).cuda(args.gpu, non_blocking=True)
        # else: real_conditions = torch.FloatTensor(real_conditions).to(device, non_blocking=True)

        # Sample noise as generator input
        z = np.random.normal(0, 1, (real_imgs.shape[0], args.latent_dim))
        if cuda_available: z = torch.cuda.FloatTensor(z).cuda(args.gpu, non_blocking=True)
        else: z = torch.FloatTensor(z).to(device, non_blocking=True)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        fake_imgs = gen_net(z, fake_conditions_reshaped).detach()
        d_loss_components = {}

        assert fake_imgs.size() == real_imgs.size(), f"fake_imgs.size(): {fake_imgs.size()} real_imgs.size(): {real_imgs.size()}"
        real_validity = dis_net(real_imgs, target_conditions)
        fake_validity = dis_net(fake_imgs, target_conditions)

        # cal loss
        if args.loss == 'hinge':
            d_loss_components['real'] = torch.mean(nn.ReLU(inplace=True)(1 - real_validity))
            d_loss_components['fake'] = torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        elif args.loss == 'standard':
            #soft label
            real_label = torch.full((real_imgs.shape[0],), 0.9, dtype=torch.float, device=device)
            fake_label = torch.full((fake_imgs.shape[0],), 0.1, dtype=torch.float, device=device)
            real_validity = nn.Sigmoid()(real_validity.view(-1))
            fake_validity = nn.Sigmoid()(fake_validity.view(-1))
            d_loss_components['real'] = nn.BCELoss()(real_validity, real_label)
            d_loss_components['fake'] = nn.BCELoss()(fake_validity, fake_label)
        elif args.loss == 'lsgan':
            if isinstance(fake_validity, list):
                d_loss_components['real'] = 0
                d_loss_components['fake'] = 0
                for real_validity_item, fake_validity_item in zip(real_validity, fake_validity):
                    real_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 1., dtype=torch.float, device=device)
                    fake_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 0., dtype=torch.float, device=device)
                    d_loss_components['real'] += nn.MSELoss()(real_validity_item, real_label)
                    d_loss_components['fake'] += nn.MSELoss()(fake_validity_item, fake_label)
            else:
                real_label = torch.full((real_validity.shape[0],real_validity.shape[1]), 1., dtype=torch.float, device=device)
                fake_label = torch.full((real_validity.shape[0],real_validity.shape[1]), 0., dtype=torch.float, device=device)
                d_loss_components['real'] = nn.MSELoss()(real_validity, real_label)
                d_loss_components['fake'] = nn.MSELoss()(fake_validity, fake_label)
        elif args.loss=='wgangp' or args.loss=='wgangp-mode':
            d_loss_components['real'] = -torch.mean(real_validity) # (torch.log(real_validity+1e-20))
            d_loss_components['fake'] = torch.mean(fake_validity)
            d_loss_components['gp'] = compute_gradient_penalty(dis_net, real_imgs, target_conditions, fake_imgs.detach(), args.phi) * 10 / (args.phi ** 2)
        elif args.loss=='wgangp-eps' or args.loss=='wgangp-mode-eps':
            d_loss_components['real'] = -torch.mean(real_validity**3)
            d_loss_components['fake'] = torch.mean(fake_validity**3)
            d_loss_components['gp'] = compute_gradient_penalty(dis_net, real_imgs, target_conditions, fake_imgs.detach(), args.phi) * 10 / (args.phi ** 2)
            d_loss_components['eps'] = (torch.mean(real_validity) ** 2) * 1e-3 # Adds eps
        else:
            raise NotImplementedError(args.loss)
        
        d_loss = 0
        for key,val in d_loss_components.items():
            d_loss_components[key] = val/float(args.accumulated_times)
            d_loss += d_loss_components[key]
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
                z = None
                if cuda_available: z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
                else: z = torch.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim))).to(device)
                fake_imgs = gen_net(z, target_conditions_reshaped)
                fake_validity = dis_net(fake_imgs, target_conditions)
                real_validity = dis_net(real_imgs, target_conditions)

                # cal loss
                if args.loss == "standard":
                    real_label = torch.full((args.gen_batch_size,), 1., dtype=torch.float, device=device)
                    fake_validity = nn.Sigmoid()(fake_validity.view(-1))
                    g_loss_components['fake'] = nn.BCELoss()(fake_validity.view(-1), real_label)
                if args.loss == "lsgan":
                    if isinstance(fake_validity, list):
                        g_loss_components['fake']
                        for fake_validity_item in fake_validity:
                            real_label = torch.full((fake_validity_item.shape[0],fake_validity_item.shape[1]), 1., dtype=torch.float, device=device)
                            g_loss_components['fake'] += nn.MSELoss()(fake_validity_item, real_label)
                    else:
                        real_label = torch.full((fake_validity.shape[0],fake_validity.shape[1]), 1., dtype=torch.float, device=device)
                        # fake_validity = nn.Sigmoid()(fake_validity.view(-1))
                        g_loss_components['fake'] = nn.MSELoss()(fake_validity, real_label)
                elif args.loss=='wgangp-mode' or args.loss=='wgangp-mode-eps':
                    g_loss_components['fake_real'] = (torch.mean(fake_validity)-torch.mean(real_validity)) ** 2

                    num_sets = 4
                    set_size = args.gen_batch_size//num_sets
                    fake_image_sets = [fake_imgs[i:i+set_size] for i in range(0, args.gen_batch_size, set_size)]
                    z_sets = [z[i:i+set_size] for i in range(0, args.gen_batch_size, set_size)]
                    condition_sets = [target_conditions[i:i+set_size] for i in range(0, args.gen_batch_size, set_size)]
                    lz = 0
                    num_pairs = 0
                    for i in range(num_sets):
                        for j in range(i+1, num_sets):
                            lz += torch.mean(torch.abs(fake_image_sets[i]-fake_image_sets[j])) / torch.mean(torch.abs(z_sets[i]-z_sets[j])) / torch.mean(torch.abs(condition_sets[i]-condition_sets[j]))
                            num_pairs += 1
                    lz /= num_pairs
                    eps = 1e-20
                    g_loss_components['mode'] = 1 / (lz + eps) * 1e-4
                else:
                    raise NotImplementedError(args.loss)
                
                g_loss = 0
                for key,val in g_loss_components.items():
                    g_loss_components[key] = val/float(args.accumulated_times)
                    g_loss += g_loss_components[key]
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
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), d_loss_components_str, g_loss.item(), g_loss_components_str, ema_beta,
                 fake_vals.mean(), fake_vals.std())
            )
            del fake_imgs
            del real_imgs
            del fake_validity
            del real_validity
            del g_loss
            del d_loss

        writer_dict['train_global_steps'] = global_steps + 1


def save_samples(args, fixed_z, fid_stat, epoch, gen_net: nn.Module, writer_dict, clean_dir=True):

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
    

def to_price_paths(logreturns):
    shape = list(logreturns.shape)
    shape[-1] += 1
    price_paths = np.ones(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]-1):
                price_paths[i,j,k+1] = price_paths[i,j,k] * np.exp(logreturns[i,j,k])
    return price_paths

def compute_similarity(real_img, fake_img):
    return 0

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