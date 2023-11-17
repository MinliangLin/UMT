This is forked from https://github.com/TencentARC/UMT

# Goal

In this project, we want to train a video highlight detection model to extract compelling short-form videos from long-form videos.

# What We Have Done



# Get Started

1. Prepare feature use `process_feature_and_label.py`

2. 

# Code Structure

1. This code take fully usage of nncore.
2. most config are using python native `dict`.
3. datasets class will be loaded by `configs/_base_/datasets/qvhighlights.py`
4. model froward api:
    - /home/ubuntu/short_form/UMT/models/model.py
    - video, audio, query, saliency, meta

## Data Format: QVHighlight

```
clip feature: (75, 512)
clip sub:
    last_hidden_state: (9, 512)
    pooler_output: (512,)
panns_feature: (75, 2048)
```
