#!/bin/bash

# python generate.py

python ../eval_scripts/evaluation.py \
    --metric similarity \
    --input_path ../data_splits/validation.json \
    --submission_path validation_output_comprehensive_few_shot_Qwen_Qwen2.5-72B-Instruct_t0.1.json \
    --threshold 0.6 