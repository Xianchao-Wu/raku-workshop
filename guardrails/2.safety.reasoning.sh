#########################################################################
# File Name: 2.safety.reasoning.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Wed Apr  8 00:06:50 2026
#########################################################################
#!/bin/bash

python -m vllm.entrypoints.openai.api_server \
    --model nvidia/Nemotron-Content-Safety-Reasoning-4B \
    --port 8001 \
    --max-model-len 4096

