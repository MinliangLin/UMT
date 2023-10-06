_base_ = [
    "../_base_/models/umt_base.py",
    "../_base_/plugins/hd.py",
    '../_base_/datasets/hotstar_highlight_865.py',
    "../_base_/schedules/100e.py",
    "../_base_/runtime.py",
]


model = dict(video_enc=dict(dims=[512, 256]), query_gen='_delete_', query_dec='_delete_')
stages = dict(epochs=10)
