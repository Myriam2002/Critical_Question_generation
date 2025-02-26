#!/bin/bash

python trial_submission/generate_openai.py

python eval_scripts/evaluation.py \
    --metric similarity \
    --input_path data_splits/sample.json \
    --submission_path trial_submission/output_gpt_4omini.json \
    --threshold 0.6 