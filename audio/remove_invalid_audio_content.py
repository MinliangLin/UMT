import pandas as pd

t = pd.read_csv("/home/ubuntu/short_form/UMT/data/hotstar_highlight_865/labels.csv")
t = t.drop(columns="Unnamed: 0")

from pathlib import Path

x = Path("data/hotstar_highlight_865/panns_features").glob("*.npz")
x = set([i.stem for i in x])

x2 = Path("data/hotstar_highlight_865/clip_features").glob("*.npz")
x2 = [i.stem for i in x2]

t2 = t[t.content_id.map(str).isin(x)]
print(len(t2))
t2[t2.content_id.map(str).isin(x2)]
print(len(t2))
t2.to_csv("data/hotstar_highlight_865/labels_valid_audio.csv", index=False)
