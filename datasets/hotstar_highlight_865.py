# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.

from pathlib import Path

import numpy as np
import torch
from nncore.dataset import DATASETS
from nncore.parallel import DataContainer
from torch.utils.data import Dataset

import pandas as pd
from sklearn import metrics


@DATASETS.register()
class HotstarHighlight865(Dataset):
    def __init__(
        self, label_path, video_path, audio_path, query_path=None, state="train"
    ):
        self.label_path = label_path
        self.video_path = video_path
        self.audio_path = audio_path
        self.query_path = query_path
        self.label = pd.read_csv(self.label_path)
        if state == "train":
            self.label = self.label[self.label.is_train].reset_index()
        else:
            self.label = self.label[~self.label.is_train].reset_index()
        is_valid = []
        for i in range(len(self.label)):
            v = self.get_video(i)
            a = self.get_audio(i)
            is_valid.append((a is not None) and (v is not None) and len(a) == len(v) > 0)
        self.label = self.label[is_valid].reset_index(drop=True)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        video = self.get_video(idx)
        audio = self.get_audio(idx)
        saliency = self.get_saliency(idx)
        row = self.label.iloc[idx]
        data = dict(
            video=DataContainer(video),
            audio=DataContainer(audio),
            saliency=DataContainer(saliency, pad_value=-1),
            info=DataContainer(
                [idx, row.content_id, row.start, row.end], cpu_only=True
            ),
        )
        return data

    def get_video(self, idx):
        row = self.label.iloc[idx]
        path = Path(self.video_path) / f"{row.content_id}.npz"
        if not path.is_file():
            return None
        video = np.load(path)["features"]
        video = video[row.start : row.end]
        return torch.from_numpy(video).float()

    def get_audio(self, idx):
        row = self.label.iloc[idx]
        path = Path(self.audio_path) / f"{row.content_id}.npz"
        if not path.is_file():
            return None
        audio = np.load(path)["arr_0"]
        audio = np.repeat(audio, 2, axis=0)
        audio = audio[row.start : row.end]
        return torch.from_numpy(audio).float()

    def get_saliency(self, idx):
        row = self.label.iloc[idx]
        duration = row.end - row.start
        # NOTE: The capitalized "Tensor" is float32 by default while "tensor" is not
        saliency = torch.Tensor([row.label] * duration)
        return saliency

    # blob: [{'meta':None, 'saliency': tensor()}]
    def evaluate(self, blob: list[dict], **kwargs):
        pred = [i["saliency"][0][0] for i in blob]
        label = self.label.label
        return {
            "AP": metrics.average_precision_score(label, pred),
            "AUC": metrics.roc_auc_score(label, pred),
            "Recall": metrics.recall_score(label, [int(i>0.5) for i in pred]),
            "Precision": metrics.precision_score(label, [int(i>0.5) for i in pred]),
        }
