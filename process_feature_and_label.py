from pathlib import Path
import shutil

# # clip video feature
# src = Path('/home/ubuntu/UniVTG/tmp')
# for i in sorted(src.glob('*/vid.npz')):
#     idx = i.parent.name.split('_')[0]
#     dst = Path('/home/ubuntu/short_form/UMT/data/hotstar_highlight_865/clip_features') / (idx + '.npz')
#     print(shutil.copy(i, dst))

# panns_feature.npz
# src = Path("/home/ubuntu/short_form/data")
# for i in sorted(src.glob("*/panns_feature.npz")):
#     idx = i.parent.name.split("_")[0]
#     dst = Path(
#         "/home/ubuntu/short_form/UMT/data/hotstar_highlight_865/panns_features"
#     ) / (idx + ".npz")
#     print(shutil.copy(i, dst))


import pandas as pd

df = pd.read_csv("/home/ubuntu/short_form/UMT/data/hotstar_highlight_865/label_ID6.csv")
df["rating"] = ""
for col in df.columns:
    if col.startswith("rating_"):
        mask = (~df[col].isna()) & (df[col] != "")
        df.loc[mask, ["rating"]] = df.loc[mask, ["rating", col]].max(axis=1)

df = df[df["rating"] != ""].reset_index(drop=True)
df["rating"] = df["rating"].str.slice(0, 1).map(int)
df["label"] = (df["rating"] > 3).map(int)
df["start"] = df["start_position_scene"].map(
    lambda x: int(pd.Timedelta(x).total_seconds())
)
df["end"] = df["end_position_scene"].map(lambda x: int(pd.Timedelta(x).total_seconds()))

# train_valid split
df["is_train"] = df.index < (len(df) * 0.9)
cols = [
    "content_id",
    "start",
    "end",
    "label",
    "rating",
    "is_train",
]
df[cols].sample(frac=1, random_state=7).to_csv(
    "/home/ubuntu/short_form/UMT/data/hotstar_highlight_865/labels.csv", index=False
)
