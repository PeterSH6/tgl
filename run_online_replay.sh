#!/bin/bash
MODEL=$1
DATA=$2
NPROC_PER_NODE=${3:-9}
NGPUS=${4:-8}


cmd="python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE \
     online_train_dist_eval_replay.py --data $DATA \
     --config /home/ubuntu/repos/tgl/config/dist/$MODEL.yml \
     --num_gpus $NGPUS"
    
rm -rf /dev/shm/*
echo $cmd
exec $cmd > accu-retrain-replay0.02-30%online-$MODEL-$DATA-$NPROC_PER_NODE-$NGPUS.log 2>&1

