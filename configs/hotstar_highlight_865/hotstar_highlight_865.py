_base_ = [
    "../_base_/models/umt_base.py",
    "../_base_/plugins/no_query.py",
    "../_base_/plugins/hd.py",
    '../_base_/datasets/hotstar_highlight_865.py',
    "../_base_/schedules/100e.py",
    "../_base_/runtime.py",
]

model = dict(video_enc=dict(dims=[512, 256]))
stages = dict(epochs=10)

