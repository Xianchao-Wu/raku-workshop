#########################################################################
# File Name: 3.grpo.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Wed Apr  8 07:52:51 2026
#########################################################################
#!/bin/bash

config=""
uv run examples/run_grpo.py \
    --config $config 
	#examples/configs/grpo_math_1B_megatron.yaml
