1. This code take fully usage of nncore.
2. most config are using python native `dict`.
3. datasets class will be loaded by `configs/_base_/datasets/qvhighlights.py`
4. model froward api:
    - /home/ubuntu/short_form/UMT/models/model.py
    - video, audio, query, saliency, meta
5. 


# QVHighlight format
clip feature: (75, 512)
clip sub:
    last_hidden_state: (9, 512)
    pooler_output: (512,)
panns_feature: (75, 2048)


