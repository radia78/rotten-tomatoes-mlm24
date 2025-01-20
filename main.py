from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning import LightningModule
import pandas as pd
import os
import torch
from torch import nn
import torch._dynamo.config
from utils.tools import encode_mask
from utils.data import TomatoLeafDataModule

class TrainingModule(LightningModule):
    def __init__(self, net: nn.Module, threshold: float=0.9):
        super().__init__()
        self.model = net
        self.threshold = threshold
        self.training_step_outputs = []
        self.val_step_outputs = []
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch["image"], batch["mask"]
        _ , loss = self.model(inputs, targets.float())
        self.training_step_outputs(loss)

        return loss
    
    def on_train_epoch_end(self):
        epoch_mean = torch.stack(self.training_step_outputs).mean()
        self.log("Training Avg Epoch Loss", epoch_mean, on_epoch=True)
        self.training_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch['image'], batch['mask']
        _, loss = self.model(inputs, targets.float())
        self.val_step_outputs.append(loss)

        return loss
    
    def on_validation_epoch_end(self):
        epoch_mean = torch.stack(self.val_step_outputs).mean()
        self.log("Validation Avg Epoch Loss", epoch_mean, on_epoch=True)
        self.val_step_outputs.clear()
    
    def predict_step(self, batch, batch_idx):
        # turned off gradients for predictions
        with torch.no_grad():
            outputs, _ = self.model(batch) 

        enc_mask = encode_mask(outputs, self.threshold)
        
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