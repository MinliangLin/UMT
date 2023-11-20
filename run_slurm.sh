#!/bin/bash
# We once tried venv on EFS, which is deprecated now.
# TODO: replace this with conda environment!
source /efs/venv/torch/bin/activate
python tools/launch.py configs/hotstar_highlight_865/hotstar_highlight_865.py
