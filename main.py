from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning import LightningModule
import pandas as pd
import os
import torch
from torch import nn
import torch._dynamo.config
from utils.tools import encode_mask
from utils.data import TomatoLeafDataModule, RetinalVesselDataModule

class TrainingModule(LightningModule):
    def __init__(self,net: nn.Module, criterion: nn.Module, threshold: float=0.9, SDL: bool=False):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.threshold = threshold
        self.training_step_outputs = []
        self.val_step_outputs = []
        self.sdl = SDL

    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        if self.sdl:
            x, y, y_hat = batch['image'], batch['mask'], batch['pseudo_mask']
            y_pred = self.forward(x)
            loss = self.criterion(y_pred, y_hat) + self.criterion(y_pred, y)

        else:
            x, y = batch["image"], batch["mask"]
            y_pred = self.forward(x)
            loss = self.criterion(y_pred, y)

        self.training_step_outputs.append(loss)
        return loss
    
    def on_train_epoch_end(self):
        epoch_mean = torch.stack(self.training_step_outputs).mean()
        self.log("Training Avg Epoch Loss", epoch_mean, on_epoch=True)
        self.training_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y_pred = self.net(x)
        loss = self.criterion(y_pred, y)
        self.val_step_outputs.append(loss)
        return loss
    
    def on_validation_epoch_end(self):
        epoch_mean = torch.stack(self.val_step_outputs).mean()
        self.log("Validation Avg Epoch Loss", epoch_mean, on_epoch=True)
        self.val_step_outputs.clear()
    
    def predict_step(self, batch, batch_idx):
        enc_mask = encode_mask(self(batch['image']), self.threshold)
        
        return {'id': batch['id'][0], 'annotation': enc_mask}

class PredictionWriter(BasePredictionWriter):
    def __init__(
            self, 
            output_dir,
            write_interval = "epoch"
        ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        pd.DataFrame(predictions).to_csv(os.path.join(self.output_dir, 'predictions.csv'))
    
def cli_main():   
    cli = LightningCLI(model_class=TrainingModule, save_config_callback=None)

if __name__ == "__main__":
    cli_main()