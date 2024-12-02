import pytorch_lightning as pl
from datamodule import OphNetOperationDataModule
from model import VideoClassificationLightningModule
import torch

gpus = [2,3] if torch.cuda.is_available() else None

def train():
    classification_module = VideoClassificationLightningModule()
    data_module = OphNetOperationDataModule(
        input_size=256,
        crop_size=244,
        mean=[0.45, 0.45, 0.45],
        std=[0.225, 0.225, 0.225],
        num_frames=8,
        fps=25,
    )
    trainer = pl.Trainer(
        devices=gpus,
        max_epochs=1000,
        strategy='ddp',
    )
    trainer.fit(model=classification_module, datamodule=data_module)

if __name__=='__main__':
    train()