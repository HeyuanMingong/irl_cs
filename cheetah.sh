#!/bin/bash

:<<BLOCK
echo -e "\nTrain in the original environment, goal velocity 1.0"
python main.py --env HalfCheetahVel-v1 --num_iter 2000 \
    --stage pretrain --batch_size 50 --algorithm trpo --task 1.0 \
    --output output/cheetah --model_path saves/cheetah \
    --num_workers 16 --device cuda:4
BLOCK

echo -e "\nThe environment changes, goal velocity 0.4"
python main.py --env HalfCheetahVel-v1 --stage finetune --task 0.4 \
    --num_iter 2000 --algorithm trpo --batch_size 50 \
    --output output/cheetah --model_path saves/cheetah \
    --random --pretrained --prpg --priw \
    --num_workers 16 --device cuda:4
