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

git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl --recursive
cd nemo-rl

# If you are already cloned without the recursive option, you can initialize the submodules recursively
git submodule update --init --recursive

# Different branches of the repo can have different pinned versions of these third-party submodules. Ensure
# submodules are automatically updated after switching branches or pulling updates by configuring git with:
# git config submodule.recurse true

# **NOTE**: this setting will not download **new** or remove **old** submodules with the branch's changes.
# You will have to run the full `git submodule update --init --recursive` command in these situations.


config="../sft/sft.yaml" # TODO change this dir if necessary
uv run python examples/run_sft.py --config $config #{config yaml}
