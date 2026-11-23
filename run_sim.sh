#!/usr/bin/env bash
# Starts the Panda simulation in Gazebo Harmonic.
# Usage: bash run_sim.sh

# Stop if any command fails.
set -e

# Run from this script's folder so the relative paths below always work.
cd "$(dirname "$0")"

# Tell Gazebo where to look for model://panda.
export GZ_SIM_RESOURCE_PATH="$PWD/models:${GZ_SIM_RESOURCE_PATH:-}"

# -r starts the physics immediately. Without it the world opens paused.
# -v 3 prints errors, warnings and general messages to this terminal.
exec gz sim -r -v 3 worlds/panda_world.sdf
