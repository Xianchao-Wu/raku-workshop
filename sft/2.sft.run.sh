#########################################################################
# File Name: 2.sft.run.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Wed Apr  8 07:43:36 2026
#########################################################################
#!/bin/bash


# git clone https://github.com/NVIDIA-NeMo/RL.git
# cd nemo-rl # TODO
# cd /workspace/asr/brev.nemo.curator.20260324/nemo-rl # for example 
# and, check examples/run_sft.py exists or not in your [nemo-rl] folder

config="sft.yaml"
uv run python examples/run_sft.py --config $config #{config yaml}
