# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.

from pathlib import Path

import nncore
import numpy as np
import torch
from nncore.dataset import DATASETS
from nncore.parallel import DataContainer
from torch.utils.data import Dataset

import pandas as pd
from .utils import eval_qvhighlights


@DATASETS.register()
class HotstarHighlight865(Dataset):
    def __init__(self, label_path, video_path, audio_path, query_path, state="train"):
        self.label_path = label_path
        self.video_path = video_path
        self.audio_path = audio_path
        self.query_path = query_path # TODO:  not implemented
        self.label = pd.read_csv(self.label_path)
        if state == "train":
            self.label = self.label[self.label.is_train].reset_index()
        else:
            self.label = self.label[~self.label.is_train].reset_index()
        self.label["start"] = self.label["start_position_scene"].map(
            lambda x: int(pd.Timedelta(x).total_seconds())
        )
        self.label["end"] = self.label["start_position_scene"].map(
            lambda x: int(pd.Timedelta(x).total_seconds())
        )

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        video = self.get_video(idx)
        audio = self.get_audio(idx)
        saliency = self.get_saliency(idx)
        data = dict(
            video=DataContainer(video),
            audio=DataContainer(audio),
            query=DataContainer(),
            saliency=DataContainer(saliency),
        )
        return data

    def get_video(self, idx):
        row = self.label.iloc[idx]
        video = np.load(Path(self.video_path) / f"{row.content_id}.npz")
        video = video[row.start:row.end]
        return torch.from_numpy(video).float()

    def get_audio(self, idx):
        row = self.label.iloc[idx]
        audio = np.load(Path(self.audio_path) / f"{row.content_id}.npz")
        audio = audio[row.start:row.end]
        return torch.from_numpy(audio).float()

    def get_saliency(self, idx):
        row = self.label.iloc[idx]
        duration = row.end - row.start
        saliency = torch.tensor([int(row.label>3)] * duration, requires_grad=False)
        return saliency

    def evaluate(self, blob, **kwargs):
        num_samples, collected = len(blob), []
        blob = nncore.to_dict_of_list(blob)

        for i in range(num_samples):
            pred = dict(qid=blob["meta"][i][0]["qid"], vid=blob["meta"][i][0]["vid"])

            if "saliency" in blob:
                pred["pred_saliency_scores"] = blob["saliency"][i][0].tolist()

            if "boundary" in blob:
                pred["pred_relevant_windows"] = blob["boundary"][i][0].tolist()

            collected.append(pred)

        label = nncore.load(self.label_path)
        results = eval_qvhighlights(collected, label)["brief"]

        return results
