#!/bin/bash

accelerate launch training/train_lora_ss.py \                                                                             INT py trellis root@kantjjwang-1eqpivo4ka 12:50:54 AM
    --config training/configs/default.yaml \
    --output_dir training/outputs/lora_experiment \
    --learning_rate 1e-4
