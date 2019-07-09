#!/bin/bash

echo "Train in the original environment, goal velocity 1.0"
python main.py --env SwimmerVel-v1 --num_iter 100 --task 1.0 \
    --stage pretrain --batch_size 50 --algorithm trpo \
    --output output/swimmer --model_path saves/swimmer \
    --num_workers 16 --device cuda:4

:<<BLOCK
echo "The environment changes, goal velocity 0.7"
python main.py --env SwimmerVel-v1 --stage finetune --trial 1 --task 0.7 \
    --num_iter 100 --relax_epis 0 --is_eps 1e-2 --batch_size 50 \
    --upsilon 1e-2 --nu 0.5 --rmax 200 --psi 1.0 --model_num 2000 \
    --output demo_output/swimmer --model_path demo_saves/swimmer \
    --ran --fine --no-isam --no-wei --iwis --prpg \
    --algorithm trpo --num_workers 4 --device cpu
BLOCK
