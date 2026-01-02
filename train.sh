#!/bin/bash

# disable core dumps
ulimit -c 0

# debugging
export HYDRA_FULL_ERROR=1
export TORCH_USE_CUDA_DSA=1
# disable for official training run
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
# export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0

# debug thread thrashing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# # Force oneDNN (MKL-DNN) to stay single-threaded on ARM
# export DNNL_MAX_CPU_RESERVATION=1
# export ONEDNN_PRIMITIVE_CACHE_CAPACITY=1024

# # Ensure NVPL/OpenBLAS doesn't over-thread
# export NVPL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1

echo "Node: $SLURM_NODEID | Rank: $SLURM_PROCID | LocalID: $SLURM_LOCALID | GPU: $CUDA_VISIBLE_DEVICES"

python3 train.py --config-name "RPF_base_main" \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    trainer.devices=$SLURM_NTASKS_PER_NODE \
    trainer.strategy="ddp_find_unused_parameters_true"\
    data_root=$DATA_DIR \
    data.batch_size=32 \
    data.num_workers=8 \
    data=breaking_bad \
    model.anchor_free=False \
    model.encoder_ckpt="./weights/RPF_base_pretrain_ep600.ckpt" \
    model.use_repa=True \
    model.optimizer.lr=1e-4 \
    model.optimizer.eps=1e-8 \
    experiment_name="ALPS_RPF_${EXPERIMENT}"

# model.optimizer.lr=5e-4
