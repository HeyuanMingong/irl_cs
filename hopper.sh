#!/bin/bash

echo "Train in the original environment, goal velocity 1.0"
python main.py --env HopperVel-v1 --save_every 2000 --num_iter 2000 \
    --stage pretrain --batch_size 50 --algorithm trpo --task 1.0 \
    --output demo_output/hopper --model_path demo_saves/hopper \
    --num_workers 4 --device cpu

echo "The environment changes, goal velocity 1.6"
python main.py --env HopperVel-v1 --stage finetune --trial 1 --task 1.6 \
    --num_iter 500 --relax_epis 0 --is_eps 1e-2 --batch_size 50 \
    --upsilon 1e-2 --nu 0.5 --rmax 200 --psi 1.0 --model_num 2000 \
    --output demo_output/hopper --model_path demo_saves/hopper \
    --ran --fine --no-isam --no-wei --iwis --prpg --no-iwinv \
    --algorithm trpo --num_workers 4 --device cpu

