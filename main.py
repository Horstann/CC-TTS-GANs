import os
import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=str, default="0")
    parser.add_argument('--node', type=str, default="0015")
    # Use the passed 'args' if provided, otherwise use default values
    return parser.parse_args(args=args)

# Here, you can manually specify the arguments as a list
args = parse_args(['--rank', '0', '--node', '0020'])
checkpoint_file = './logs/wgangp-mode/Model/checkpoint'
batch_size = 8
# --load_path {checkpoint_file} \
# '/kaggle/working/logs'

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.chdir(r'C:\Users\Horstann\Documents\NTU URECA\tts-gan-main')
os.system(
    f"python train_GAN.py \
    -gen_bs {batch_size} \
    -dis_bs {batch_size} \
    --batch_size {batch_size} \
    --dist-url 'tcp://localhost:4321' \
    --dist-backend 'nccl' \
    --world-size 1 \
    --rank {args.rank} \
    --log_dir ./logs \
    --dataset UniMiB \
    --bottom_width 8 \
    --max_iter 500000 \
    --img_size 32 \
    --gen_model my_gen \
    --dis_model my_dis \
    --df_dim 384 \
    --d_heads 4 \
    --d_depth 3 \
    --g_depth 5,4,2 \
    --dropout 0 \
    --latent_dim 100 \
    --gf_dim 1024 \
    --num_workers 8 \
    --g_lr 0.0001 \
    --d_lr 0.0003 \
    --optimizer adam \
    --loss wgangp-mode \
    --wd 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --phi 1 \
    --num_eval_imgs 50000 \
    --init_type xavier_uniform \
    --n_critic 1 \
    --val_freq 20 \
    --print_freq 50 \
    --grow_steps 0 0 \
    --fade_in 0 \
    --patch_size 2 \
    --ema_kimg 500 \
    --ema_warmup 0.1 \
    --ema 0.9999 \
    --diff_aug translation,cutout,color \
    --exp_name Running"
)