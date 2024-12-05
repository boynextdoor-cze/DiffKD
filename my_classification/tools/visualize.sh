#!/bin/bash
MASTER_PORT=29500
export PYTHONPATH=$(pwd):$PYTHONPATH

set -x

# NOTE: This script only supports run on single machine and single (multiple) GPUs.
#       You may need to modify it to support multi-machine multi-card training on your distributed platform.

python -m torch.distributed.launch --nproc_per_node=1 tools/visualize.py -c configs/strategies/distill/diffkd/diffkd_a1.yaml --model cifar_ShuffleV1 --experiment visualization 
