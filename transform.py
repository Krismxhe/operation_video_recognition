from torchvision.transforms import Compose, Lambda

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    RandomCrop,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo,
    RandomShortSideScale,
    RandomResizedCrop,
)

from typing import Dict, List, Optional

def val_transform(
    input_size: int,
    crop_size: int,
    mean: List[float],
    std: List[float],
    num_frames: int,
    fps: Optional[int] = None,
    ):

    """
    Transforms to sampled clips
    """

    side_size = input_size
    crop_size = crop_size
    mean = mean
    std = std

    # input frame seq length
    num_frames = num_frames

    transform = ApplyTransformToKey(
        key="video",
        transform = Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size),
                CenterCropVideo(
                    crop_size=(crop_size, crop_size)
                )
            ]
        )
    )

    return transform

def x3d_train_transform(
    input_size: int,
    crop_size: int,
    mean: List[float],
    std: List[float],
    num_frames: int,
    fps: Optional[int] = None,
    ):

    side_size = input_size
    crop_size = crop_size
    mean = mean
    std = std

    # input frame seq length
    num_frames = num_frames

    train_transform = ApplyTransformToKey(
        key='video',
        transform = Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                RandomShortSideScale(min_size=input_size, max_size=320),
                RandomCrop(crop_size),
            ]
        )
    )

    return train_transform
