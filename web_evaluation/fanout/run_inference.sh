#!/bin/bash
set -x

python inference_fanout.py \
    --agent OnepassAgent \
    --port 3000 \
    --model ft-Meta-Llama-3-8B-Instruct \
    overfit_test
