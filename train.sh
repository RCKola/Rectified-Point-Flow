#!/bin/bash

echo "Node: $SLURM_NODEID | Rank: $SLURM_PROCID | LocalID: $SLURM_LOCALID | GPU: $CUDA_VISIBLE_DEVICES"

python3 train.py --config-name "RPF_base_main" \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    trainer.devices=$SLURM_NTASKS_PER_NODE \
    trainer.strategy="ddp_find_unused_parameters_true"\
    trainer.precision="bf16" \
    data_root=$DATA_DIR \
    data.batch_size=32 \
    data.num_workers=8 \
    data=breaking_bad \
    model.anchor_free=False \
    model.encoder_ckpt="./weights/RPF_base_pretrain_ep600.ckpt" \
    model.use_repa=True \
    model.optimizer.eps=1e-6 \
    experiment_name="ALPS_RPF_${EXPERIMENT}"

# model.optimizer.lr=5e-4
