#!/bin/bash

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate herdnet

# tiling data
python /home/ubuntu/workspace/HerdNet/tools/patcher.py \
       /home/ubuntu/workspace/EBP-Lindanda-cam0 2048 2048 128 \
       /home/ubuntu/workspace/EBP-Lindanda-cam0-splits -min 0.0 -all True