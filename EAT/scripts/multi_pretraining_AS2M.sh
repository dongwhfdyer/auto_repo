#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

python fairseq_cli/hydra_train.py -m \
    --config-dir EAT/config \
    --config-name multi_pretraining_AS2M \
    common.user_dir=EAT \
    checkpoint.save_dir=/hpc_stor03/sjtu_home/wenxi.chen/model_ckpt/multi_pretrain \
    checkpoint.restore_file=/hpc_stor03/sjtu_home/wenxi.chen/model_ckpt/multi_pretrain/checkpoint_last.pt \
    distributed_training.distributed_world_size=4 \
    dataset.batch_size=12 \
    task.old_data=/path/to/old/data \
    task.new_data=/path/to/new/data \
    task.roll_aug=true \
    task.noise=true 