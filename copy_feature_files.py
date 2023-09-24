from pathlib import Path
import shutil

# # clip video feature
# src = Path('/home/ubuntu/UniVTG/tmp')
# for i in sorted(src.glob('*/vid.npz')):
#     idx = i.parent.name.split('_')[0]
#     dst = Path('/home/ubuntu/short_form/UMT/data/hotstar_highlight_865/clip_features') / (idx + '.npz')
#     print(shutil.copy(i, dst))

# panns_feature.npz
src = Path("/home/ubuntu/short_form/data")
for i in sorted(src.glob("*/panns_feature.npz")):
    idx = i.parent.name.split("_")[0]
    dst = Path(
        "/home/ubuntu/short_form/UMT/data/hotstar_highlight_865/panns_features"
    ) / (idx + ".npz")
    print(shutil.copy(i, dst))


import pandas as pd

df = pd.read_csv("/home/ubuntu/short_form/UMT/data/hotstar_highlight_865/label_ID6.csv")
df['rating'] = ''
for col in df.columns:
    if col.startswith("rating_"):
        mask = (~df[col].isna()) & (df[col] != "")
        df['rating'][mask] = df[['rating', col]][mask].max(axis=1)

cols = ['content_id', 'start_position_scene', 'end_position_scene', 'rating', 'is_train']
df = df[df['rating']!=''].reset_index(drop=True)
df['is_train'] = df.index < (len(df) * 0.9)
df[cols].to_csv("/home/ubuntu/short_form/UMT/data/hotstar_highlight_865/labels.csv", index=False)
