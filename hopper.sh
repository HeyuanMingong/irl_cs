#!/bin/bash

echo -e "\nTrain in the original environment, goal velocity 1.0"
python main.py --env HopperVel-v1 --num_iter 500 --task 1.0 \
    --stage pretrain --batch_size 50 --algorithm trpo \
    --output output/hopper --model_path saves/hopper \
    --num_workers 16 --device cuda:0

echo -e "\nThe environment changes, goal velocity 1.6"
python main.py --env HopperVel-v1 --stage finetune --task 1.6 \
    --num_iter 500  --batch_size 50 --algorithm trpo \
    --output output/hopper --model_path saves/hopper \
    --random --pretrained --prpg --priw \
    --num_workers 16 --device cuda:0
