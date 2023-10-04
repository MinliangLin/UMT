_base_ = [
    "_base_/models/umt_base.py",
    "_base_/plugins/no_query.py",
    "_base_/plugins/hd.py",
    "_base_/schedules/100e.py",
    "_base_/runtime.py",
    "../datasets/hotstar_highlight_865.py",
]

model = dict(video_enc=dict(dims=[512, 256]))
stages = dict(epochs=10)

# dataset settings
dataset_type = "HotstarHighlight865"
data_root = "data/hotstar_highlight_865/"
data = dict(
    train=dict(
        type=dataset_type,
        label_path=data_root + "labels.csv",
        video_path=data_root + "clip_features",
        audio_path=data_root + "panns_features",
        loader=dict(batch_size=32, num_workers=4, shuffle=True),
        state='train',
    ),
    val=dict(
        type=dataset_type,
        label_path=data_root + "labels.csv",
        video_path=data_root + "clip_features",
        audio_path=data_root + "panns_features",
        loader=dict(batch_size=1, num_workers=4, shuffle=False),
        state='val',
    ),
)
