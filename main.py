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

# seq_len = 10
# patch_size = 2 # patch_size must be a factor of seq_len
# dis_emb_size = 5 # from (num_patches, patch_size) to (num_patches, emb_size)
# gen_emb_size = 10 # from (latent_dim,) to (seq_len, gen_emb_size))
# var_term_weight = 1e-3

seq_len = 42
patch_size = 6 
assert seq_len%patch_size == 0
dis_emb_size = 50 # from (num_patches, patch_size) to (num_patches, emb_size)
gen_emb_size = 10 # from (latent_dim,) to (seq_len, gen_emb_size))
m2x_term_weight = 1e-02
var_term_weight = 0

batch_size = 4
gen_sample_size = 30
dis_sample_size = 10
assert gen_sample_size >= dis_sample_size
max_epoch = 40
n_critic = 1
# --load_path {checkpoint_file} \
# '/kaggle/working/logs'

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.chdir(r'C:\Users\Horstann\Documents\NTU URECA\tts-gan-main')
os.system(
    f"python train_GAN.py \
    -gen_bs {batch_size} \
    -dis_bs {batch_size} \
    --batch_size {batch_size} \
    --gen_sample_size {gen_sample_size} \
    --dis_sample_size {dis_sample_size} \
    --seq_len {seq_len} \
    --patch_size {patch_size} \
    --gen_emb_size {gen_emb_size} \
    --dis_emb_size {dis_emb_size} \
    --m2x_term_weight {m2x_term_weight} \
    --var_term_weight {var_term_weight} \
    --max_epoch {max_epoch} \
    --dist-url 'tcp://localhost:4321' \
    --dist-backend 'nccl' \
    --world-size 1 \
    --rank {args.rank} \
    --log_dir ./logs \
    --dataset UniMiB \
    --bottom_width 8 \
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
    --ema_kimg 500 \
    --ema_warmup 0.1 \
    --ema 0.9999 \
    --diff_aug translation,cutout,color \
    --exp_name model"
)