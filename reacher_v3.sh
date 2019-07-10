#!/bin/bash

### train in the original environment, task: goal [0.1,0.1], with reacher.xml
echo -e "\nTrain in the original environment, goal [0.1, 0.1], with reacher_0.xml"
python main.py --env ReacherDyna-v3 --task 0.1 0.1 0 --num_iter 500 \
    --lr 0.01 --algorithm reinforce --stage pretrain --batch_size 50 \
    --output output/reacher_v3 --model_path saves/reacher_v3 \
    --num_workers 4 --device cpu


echo -e "\nThe environment changes to a new one as [0, 0.14] with reacher_5.xml"
python main.py --env ReacherDyna-v3 --stage finetune --task 0.14 0 5 \
    --num_iter 500 --batch_size 50 --algorithm reinforce --lr 0.01 \
    --output output/reacher_v3 --model_path saves/reacher_v3 \
    --random --pretrained --prpg --priw \
    --num_workers 4 --device cpu

