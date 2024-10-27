from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
from dataclasses import dataclass
import torch
import pickle
import os

class BaseTrainingSession:
    def __init__(self, args):
        """
        args: Arguments/components for the training session see the training config for details of args
        """
        # Data lodaer
        self.dataloader = args.dataloader

        # Load the necessary components
        self.loss_fn = args.loss_fn

        # Load the model
        self.model = args.model
        self.model.to(args.device)

        # Load the model name
        self.model_name = args.model_name

        # Configured optimizer and scheduler
        self.scheduler = args.scheduler
        self.optimizer = args.optimizer

        # Cache arguments
        self.grad_accumulation = args.grad_accumulation
        self.grad_acc_steps = args.grad_acc_steps
        self.save_grad = args.save_grad
        self.device = args.device
        self.total_steps = None

        # Mark the training session
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if not os.path.exists("log/tb"):
            os.makedirs("log/tb")

        # Create the tensorboard writer
        self.writer = SummaryWriter(f"log/tb/leaf_segment_training_{self.timestamp}") 

    def collect_gradients(self, img_idx):
        """
        Collect the gradient computed for a specified index for analysis
        """
        # Create a folder to cache gradients
        if not os.path.exists("grads"):
            os.makedirs("grads")
        
        grad_dict = {}
        for name, param in (self.model.named_parameters()):
            if param.requires_grad and len(param.shape) > 1:
                grad_dict[f"{name}"] = param.grad

        with open(f'grads/grad_{img_idx}.pkl', 'wb') as file:
            pickle.dump(grad_dict, file)
    
    def train_one_epoch(self, epoch_index):
        print(f"EPOCH {epoch_index}:")

        # Running loss statistic
        running_loss = 0.0

        # Empty out the gradients
        self.optimizer.zero_grad()

        for i, data in enumerate(self.dataloader):
            img, mask = data['image'], data['mask']

            # Attach the data to the device
            img, mask = img.to(self.device), mask.to(self.device)

            # Generate the logits of the predicted and true masks
            preds = self.model.forward(img)

            # Generate the loss and append it to running statistics
            loss = self.loss_fn(y_pred=preds, y_true=mask)
            running_loss += loss
            loss = loss / self.grad_acc_steps

            # Backprop
            loss.backward()

            # Collect the gradients and save it for analysis for the last iteration and epoch
            if self.save_grad and (epoch_index * (i + 1) == self.total_steps):
                self.collect_gradients(i)

            if self.grad_accumulation and ((i + 1) % self.grad_acc_steps == 0 or (i + 1)==len(self.dataloader)):
                # Adjust the gradients and update the learning rate
                self.optimizer.step()
                self.scheduler.step()

                # Empty out gradients
                self.optimizer.zero_grad()

            # Log the statistics
            print(f"    Batch {i} | Loss: {loss * self.grad_acc_steps}")
            self.writer.add_scalar('Loss/Train', loss, epoch_index * (i + 1))

            # Delete the objects and caches to save memory
            del loss, preds, mask
            self.empty_device_cache()

        avg_loss = running_loss/(i + 1)

        return avg_loss
    
    def empty_device_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
        else:
            return None

    def train(self, num_epochs):
        self.total_steps = num_epochs * len(self.dataloader)
        for epoch in range(1, num_epochs + 1):
            # Set to training mode
            self.model.train(True)

            # Go through the entire dataset once
            avg_loss = self.train_one_epoch(epoch)

            # Log the statistics
            print(f"Train Loss: {avg_loss}")
            self.writer.add_scalar("Average Training Loss", avg_loss, epoch)
            
            # Clear the writer
            self.writer.flush()
        
        # Save the model
        self.save_model()

    def save_model(self):
        if not os.path.exists(f"model_checkpoint/{self.model_name}"):
            os.makedirs(f"model_checkpoint/{self.model_name}")
            
        model_path = f"model_checkpoint/{self.model_name}/model_{self.timestamp}.pt"
        torch.save(self.model.state_dict(), model_path)

@dataclass
class BaseTrainingSessionConfig:
    dataloader: DataLoader
    model: torch.nn.Module
    model_name: str
    loss_fn: any
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    device: str
    grad_accumulation: bool
    grad_acc_steps: int
    save_grad: bool