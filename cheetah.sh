#!/bin/bash

echo "Train in the original environment, goal velocity 1.0"
python main.py --env HalfCheetahVel-v1 --save_every 50000 --num_iter 50000 \
    --stage pretrain --batch_size 50 --algorithm trpo --task 1.0 \
    --output demo_output/cheetah --model_path demo_saves/cheetah \
    --num_workers 4 --device cpu

echo "The environment changes, goal velocity 0.4"
python main.py --env HalfCheetahVel-v1 --stage finetune --trial 1 --task 0.4 \
    --num_iter 2000 --relax_epis 0 --is_eps 1e-2 --batch_size 50 \
    --upsilon 1e-2 --nu 0.5 --rmax 200 --psi 1.0 --model_num 50000 \
    --output demo_output/cheetah --model_path demo_saves/cheetah \
    --ran --fine --no-isam --no-wei --iwis --prpg \
    --algorithm trpo --num_workers 4 --device cpu

