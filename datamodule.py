import os
from tqdm import tqdm
from typing import Dict, List, Optional
import torch
import pytorch_lightning as pl
import pytorchvideo.data
from pytorchvideo.data import (
    labeled_video_dataset,
    UniformClipSampler,
    RandomClipSampler,
)

from transform import x3d_train_transform, val_transform 

class OphNetOperationDataModule(pl.LightningDataModule):
    # Dataset configuration
    _DATA_PATH = '/example/'
    _CLIP_DURATION = 2
    _BATCH_SIZE = 12
    _NUM_WORKERS = 4

    def __init__(
        self,
        input_size: int,
        crop_size: int,
        mean: List[float],
        std: List[float],  
        num_frames: int,
        fps: int,
    ):
        super().__init__()
        
        # attributes of input
        self.input_size = input_size
        self.crop_size = crop_size
        self.mean = mean
        self.std = std

        # attributes of video
        self.fps = fps

        # attributes of model, throughput
        self.num_frames = num_frames

    def train_dataloader(self):
        """
        Create the operation train partition from the list of video labels
        in {self._DATA_PATH} / train
        """
        clip_randsampler = RandomClipSampler(
            clip_duration = self._CLIP_DURATION
        )
        transforms = x3d_train_transform(
            input_size=self.input_size,
            crop_size=self.crop_size,
            mean=self.mean,
            std=self.std,
            num_frames=self.num_frames,
        )
        train_dataset = labeled_video_dataset(
            data_path=os.path.join(self._DATA_PATH, 'labels/train.txt'), 
            clip_sampler=clip_randsampler, 
            transform=transforms,
            video_path_prefix=os.path.join(self._DATA_PATH, 'video'),
            decode_audio=False,
            decoder="pyav"
        )

        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            pin_memory=True
        )

    def val_dataloader(self):
        clip_unisampler = UniformClipSampler(
            clip_duration=self._CLIP_DURATION,
            stride=self._CLIP_DURATION,
            backpad_last=True,
            eps=1e-06,
        )
        transforms = val_transform(
            input_size=self.input_size,
            crop_size=self.crop_size,
            mean=self.mean,
            std=self.std,
            num_frames=self.num_frames,
        )
        val_dataset = labeled_video_dataset(
            data_path=os.path.join(self._DATA_PATH, 'labels/val.txt'),
            video_path_prefix=os.path.join(self._DATA_PATH, 'video'),
            clip_sampler=clip_unisampler,
            decode_audio=False,
            transform=transforms
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            pin_memory=True
        )

if __name__ == "__main__":
    dm = OphNetOperationDataModule(
        input_size=256,
        crop_size=244,
        mean=[0.45, 0.45, 0.45],
        std=[0.225, 0.225, 0.225],
        num_frames=8,
        fps=25,
    )

    dataloader = dm.val_dataloader()

    for idx, batch in enumerate(tqdm(dataloader)):
        print(idx)
        print(batch['video_name'])

    dataloader = dm.train_dataloader()
    
    for idx, batch in enumerate(tqdm(dataloader)):
        print(idx)
        print(batch['video_name'])