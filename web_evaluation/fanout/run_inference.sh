#!/bin/bash
set -x

# python inference_fanout.py \
#     --agent PolicyAgent \
#     --port 3000 \
#     --model modularized-group-8B \
#     --n_processes 1 \
#     overfit_test

python inference_fanout.py \
    --agent PolicyAgent \
    --port 3000 \
    --model modularized-group-8B \
    --n_processes 1 \
    overfit_test
