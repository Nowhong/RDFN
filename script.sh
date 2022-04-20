CUDA_VISIBLE_DEVICES=0 python train.py --opt_path option_R3_mfqev2_1G.yml

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12390 train.py --opt_path option_R3_mfqev2_2G.yml
