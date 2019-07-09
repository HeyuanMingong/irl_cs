#!/bin/bash

### train in the original environment, task: goal [0.1,0.1], with reacher.xml
echo "Training in the original environment, goal [0.1, 0.1], with reacher_0.xml"
python main.py --env ReacherDyna-v3 --task 0.1 0.1 0 --num_iter 500 \
    --lr 0.01 --algorithm reinforce --stage pretrain --batch_size 50 \
    --output output/reacher_v3 --model_path saves/reacher_v3 \
    --num_workers 16 --device cuda:4

:<<BLOCK
echo "The environment changes to a new one as [0, 0.14] with reacher_5.xml, first trial"
python main.py --env ReacherDyna-v3 --stage finetune --trial 1 --task 0.14 0 5 \
    --num_iter 500 --relax_epis 0 --is_eps 1e-2 --batch_size 50 --model_num 3000 \
    --upsilon 1e-2 --nu 0.8 --rmax 200 --psi 1.0 --algorithm reinforce --lr 0.01 \
    --output demo_output/reacher_v3 --model_path demo_saves/reacher_v3 \
    --ran --fine --no-isam --no-wei --iwis --prpg \
    --num_workers 4 --device cpu
BLOCK
