import pandas as pd
import numpy as np
import ffmpeg


df = pd.read_csv("data/hotstar_highlight_865/labels_valid_audio.csv")
df["content_id"] = df["content_id"].astype(str)

# df['start'] = df["start_position_scene"].map(
#     lambda x: int(pd.Timedelta(x).total_seconds())
# )
# df['end'] = df["end_position_scene"].map(
#     lambda x: int(pd.Timedelta(x).total_seconds())
# )
# df['duration']=df['end']-df['start']

df = df[["content_id"]].drop_duplicates()

def get_duration(x):
    try:
        return float(ffmpeg.probe(f"/home/ubuntu/short_form/data/{x}/{x}.mp4")["format"][
            "duration"
        ])
    except Exception as e:
        return -1


df["total_duration"] = df["content_id"].map(get_duration)
df["vlen"] = df["content_id"].map(
    lambda x: np.load(f"data/hotstar_highlight_865/clip_features/{x}.npz")[
        "features"
    ].shape[0]
)
df["alen"] = df["content_id"].map(
    lambda x: np.load(f"data/hotstar_highlight_865/panns_features/{x}.npz")[
        "arr_0"
    ].shape[0]*2
)
