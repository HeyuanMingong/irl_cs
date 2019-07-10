#!/bin/bash

echo -e "\nTrain in the original environment, goal velocity 1.0"
python main.py --env SwimmerVel-v1 --num_iter 100 --task 1.0 \
    --stage pretrain --batch_size 50 --algorithm trpo \
    --output output/swimmer --model_path saves/swimmer \
    --num_workers 16 --device cuda:0


echo -e "\nThe environment changes, goal velocity 0.7"
python main.py --env SwimmerVel-v1 --stage finetune --task 0.7 \
    --num_iter 100  --batch_size 50 --algorithm trpo \
    --output output/swimmer --model_path saves/swimmer \
    --random --pretrained --prpg --priw \
    --num_workers 16 --device cuda:0

