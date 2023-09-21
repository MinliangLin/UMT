_base_ = [
    "_base_/models/umt_base.py",
    "_base_/plugins/mrhd.py",
    "_base_/schedules/200e.py",
    "_base_/runtime.py",
    "../datasets/hotstar_highlight_865.py",
]

# dataset settings
dataset_type = "QVHighlights"
data_root = "data/hotstar_highlight_865/"
data = dict(
    train=dict(
        type=dataset_type,
        label_path=data_root + "labels.csv",
        video_path=data_root + "clip_features",
        audio_path=data_root + "panns_features",
        query_path=data_root + "clip_text_features",
        loader=dict(batch_size=32, num_workers=4, shuffle=True),
        dtype='train',
    ),
    val=dict(
        type=dataset_type,
        label_path=data_root + "labels.csv",
        video_path=data_root + "clip_features",
        audio_path=data_root + "panns_features",
        query_path=data_root + "clip_text_features",
        loader=dict(batch_size=1, num_workers=4, shuffle=False),
        dtype='val',
    ),
)
