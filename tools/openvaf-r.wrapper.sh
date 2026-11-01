#!/bin/sh
# OpenVAF-Reloaded wrapper (LLVM 18 runtime from eda env)
LD_LIBRARY_PATH=/home/mahmoudasfour/miniforge3/envs/eda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} exec /home/mahmoudasfour/sim_workspace/tools/openvaf/openvaf-r-v24.0.2mob-linux-x86_64/bin/openvaf-r "$@"
