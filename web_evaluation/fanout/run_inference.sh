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

python inference_fanout.py \
    fanout-0-5-agent-model-vanilla-gpt-4o \
    --agent AgentModelAgent \
    --model gpt-4o \
    --n_processes 3 \
    --port 5000 \
    --api_key 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
