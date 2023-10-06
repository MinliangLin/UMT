_base_ = 'datasets'
# dataset settings
dataset_type = "HotstarHighlight865"
data_root = "data/hotstar_highlight_865/"
data = dict(
    train=dict(
        type=dataset_type,
        label_path=data_root + "labels_valid_audio.csv",
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
