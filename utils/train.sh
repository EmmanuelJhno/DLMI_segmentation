#!/bin/bash
batch_sizes=(2 3 4 8)
for bs in "${batch_sizes[@]}"; do
  python ./train.py --config_file ../configs/"$1/bs_$bs.json" --logdir ../runs/"$1" --num_workers 8 --seed 21
  sleep 4
done
