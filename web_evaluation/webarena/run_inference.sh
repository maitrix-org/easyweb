#!/bin/bash
set -x

BASE_URL="http://ec2-3-84-138-66.compute-1.amazonaws.com"

export SHOPPING="$BASE_URL:7770/"
export SHOPPING_ADMIN="$BASE_URL:7780/admin"
export REDDIT="$BASE_URL:9999"
export GITLAB="$BASE_URL:8023"
export WIKIPEDIA="$BASE_URL:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export MAP="$BASE_URL:3000"
export HOMEPAGE="$BASE_URL:4399"

# export AGENT_SELECTION="webarena_noplan"
# export AGENT_SELECTION="webarena_plan"

# cd ~
# bash reset_webarena_host.sh
# bash run_webarena_host.sh
# sleep 180
cd /home/ubuntu/web-agent-application
poetry run python web_evaluation/webarena/inference_webarena.py \
    --agent-cls BrowsingAgent \
    --eval-output-dir web_evaluation/webarena/results/browsingagent-full \
    --llm-config llm \
    --model gpt-4o \
    --max-iterations 15 \
    --max-retry 2 \
    --eval-num-workers 4

# cd ~
# bash reset_webarena_host.sh
# bash run_webarena_host.sh
# sleep 180
# cd /home/ubuntu/web-agent-application
# export AGENT_SELECTION="webarena_noplan"
# poetry run python web_evaluation/webarena/inference_webarena.py \
#     --agent-cls ModularWebAgent \
#     --eval-output-dir web_evaluation/webarena/results/singlepolicy-full \
#     --llm-config llm \
#     --model gpt-4o \
#     --max-iterations 15 \
#     --max-retry 2 \
#     --eval-num-workers 4

# cd ~
# bash reset_webarena_host.sh
# bash run_webarena_host.sh
# sleep 180
# cd /home/ubuntu/web-agent-application
# export AGENT_SELECTION="webarena_plan"
# poetry run python web_evaluation/webarena/inference_webarena.py \
#     --agent-cls ModularWebAgent \
#     --eval-output-dir web_evaluation/webarena/results/wmp-full \
#     --llm-config llm \
#     --model gpt-4o \
#     --max-iterations 15 \
#     --max-retry 2 \
#     --eval-num-workers 4



# cd ~
# bash reset_webarena_host.sh
# bash run_webarena_host.sh
# sleep 180
# cd /home/ubuntu/web-agent-application
# poetry run python web_evaluation/webarena/inference_webarena.py \
#     --agent-cls BrowsingAgent \
#     --eval-output-dir web_evaluation/webarena/results/browsingagent-rand300-7 \
#     --llm-config llm \
#     --model gpt-4o \
#     --eval-n-limit 300 \
#     --shuffle \
#     --seed 21 \
#     --max-iterations 15 \
#     --max-retry 2 \
#     --eval-num-workers 4

# cd ~
# bash reset_webarena_host.sh
# bash run_webarena_host.sh
# sleep 180
# cd /home/ubuntu/web-agent-application
# export AGENT_SELECTION="webarena_noplan"
# poetry run python web_evaluation/webarena/inference_webarena.py \
#     --agent-cls ModularWebAgent \
#     --eval-output-dir web_evaluation/webarena/results/singlepolicy-rand300-7 \
#     --llm-config llm \
#     --model gpt-4o \
#     --eval-n-limit 300 \
#     --shuffle \
#     --seed 21 \
#     --max-iterations 15 \
#     --max-retry 2 \
#     --eval-num-workers 4

# cd ~
# bash reset_webarena_host.sh
# bash run_webarena_host.sh
# sleep 180
# cd /home/ubuntu/web-agent-application
# export AGENT_SELECTION="webarena_plan"
# poetry run python web_evaluation/webarena/inference_webarena.py \
#     --agent-cls ModularWebAgent \
#     --eval-output-dir web_evaluation/webarena/results/wmp-rand300-7 \
#     --llm-config llm \
#     --model gpt-4o \
#     --eval-n-limit 300 \
#     --shuffle \
#     --seed 21 \
#     --max-iterations 15 \
#     --max-retry 2 \
#     --eval-num-workers 4
