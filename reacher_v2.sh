#!/bin/bash

### train in the original environment, with physical parameters reacher.xml
echo "Training in the original environment with reacher_0.xml"
python main.py --env ReacherDyna-v2 --save_every 3000 --num_iter 3000 \
    --lr 0.01 --algorithm reinforce --stage pretrain --batch_size 50 \
    --output demo_output/reacher_v2 --model_path demo_saves/reacher_v2 \
    --task 0 --num_workers 4 --device cpu

#######################################################################
echo "The environment changes to a new one as reacher_2.xml, first trial"
python main.py --env ReacherDyna-v2 --stage finetune --trial 1 --task 2 \
    --num_iter 500 --relax_epis 0 --is_eps 1e-2 --batch_size 50 --model_num 3000 \
    --upsilon 1e-2 --nu 0.8 --rmax 200 --psi 1.0 --algorithm reinforce --lr 0.01 \
    --output demo_output/reacher_v2 --model_path demo_saves/reacher_v2 \
    --ran --fine --no-isam --no-wei --iwis --prpg \
    --num_workers 4 --device cpu

