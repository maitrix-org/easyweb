#!/bin/bash
set -x

# python inference_fanout.py \
#     --agent PolicyAgent \
#     --port 3000 \
#     --model modularized-group-8B \
#     --n_processes 1 \
#     overfit_test

python inference_fanout.py \
    --agent WebPlanningAgent \
    --port 5000 \
    --model gpt-4o \
    --api_key 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 \
    --n_processes 1 \
    small_test
