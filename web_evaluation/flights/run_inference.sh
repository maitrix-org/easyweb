#!/bin/bash
set -x

python inference_flight.py \
    --agent PolicyAgent \
    --port 3000 \
    --model modularized-group-8B \
    --n_processes 1 \
    overfit_test
