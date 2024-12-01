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
    def __init__(self,net: nn.Module, criterion: nn.Module, threshold: float, point_refine: bool):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.threshold = threshold
        self.point_refine = point_refine
        if point_refine:
            self.refine_criterion = nn.BCEWithLogitsLoss()
        self.training_step_outputs = []
        self.val_step_outputs = []

    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # point refinement logic
        if self.point_refine:
            x, y = batch["image"], batch['mask']
            N = x.shape[0]
            idx, coarse_masks, refined_points = self.forward(x)
            K = idx.shape[-1]
            coarse_loss = self.criterion(coarse_masks, y)
            refine_targets = y.view(N, -1)[torch.arange(N).unsqueeze(1), idx].view(N, K)
            refine_loss = self.refine_criterion(refined_points.view(N, K), refine_targets.float())
            loss = coarse_loss + refine_loss
        
        # UNet training logic
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

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        pd.DataFrame(predictions).to_csv(os.path.join(self.output_dir, 'predictions.csv'))
    
def cli_main():   
    cli = LightningCLI(model_class=TrainingModule, save_config_callback=None)

if __name__ == "__main__":
    cli_main()