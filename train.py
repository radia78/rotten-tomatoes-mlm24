from unet.model import *
from utils.data_loading import TomatoLeafDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from argparse import ArgumentParser
import os

# DIRS
TRAIN_DIR = "data"

class TrainSession:
    def __init__(self, args):

        self.train_loader = DataLoader(
            TomatoLeafDataset(TRAIN_DIR + "/train.csv", TRAIN_DIR + "/train"), 
            batch_size=1, 
            shuffle=True, 
            num_workers=args.num_workers
        )

        # Load the necessary components
        self.loss_fn = smp.losses.JaccardLoss(smp.losses.BINARY_MODE, from_logits=True, log_loss=False)

        # Load the model
        self.model = TomatoLeafModel()
        self.model.to(args.device)

        self.T_max = args.epoch * len(self.train_loader)

        self.scheduler, self.optimizer = self.configure_optimizer(self.T_max)

        # Create a dictionary to hold the statistics
        self.metrics_dict = {
            "tp": [],
            "fp": [],
            "fn": [],
            "tn": []
        }

        # Cache arguments
        self.threshold = args.threshold

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if not os.path.exists("log/tb"):
            os.makedirs("log/tb")

        self.writer = SummaryWriter(f"log/tb/leaf_segment_training_{self.timestamp}")

    def configure_optimizer(self, T_max, lr=2e-4, eta_min=1e-5):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        return scheduler, optimizer
    
    def train_one_epoch(self, epoch_index):
        print(f"EPOCH {epoch_index}:")

        running_loss = 0.0
        for i, data in enumerate(self.train_loader):
            img, mask = data['image'], data['mask']

            # Empty out the gradients
            self.optimizer.zero_grad()

            # Generate the logits of the predicted and true masks
            logits_pred_mask = self.model.forward(img)

            # Generate the statistics loss and Jacard Index
            loss = self.loss_fn(y_pred=logits_pred_mask, y_true=mask)

            # Backprop
            loss.backward()

            # Adjust the gradients and update the learning rate
            self.optimizer.step()
            self.scheduler.step()

            tp, fp, fn, tn = smp.metrics.get_stats(
                (logits_pred_mask.sigmoid() > self.threshold).long(), mask.long(), mode="binary"
            )

            running_loss += loss

            # Log the statistics
            print(f"    Image {i} | Loss: {loss}")
            self.writer.add_scalar('Loss/Train', loss, epoch_index * (i + 1))

            # Append the running stats
            self.metrics_dict['tp'].append(tp)
            self.metrics_dict['fp'].append(fp)
            self.metrics_dict['fn'].append(fn)
            self.metrics_dict['tn'].append(tn)
        
        per_image_iou = smp.metrics.iou_score(
            torch.cat([x for x in self.metrics_dict['tp']]),
            torch.cat([x for x in self.metrics_dict['fp']]),
            torch.cat([x for x in self.metrics_dict['fn']]),
            torch.cat([x for x in self.metrics_dict['tn']]),
            reduction="micro-imagewise"
        )

        avg_loss = running_loss/(i + 1)

        # Clear cache
        for key in self.metrics_dict:
            self.metrics_dict[key].clear()

        return avg_loss, per_image_iou

    def train(self, num_epochs):

        for epoch in range(1, num_epochs + 1):
            # Set to training mode
            self.model.train(True)

            # Go through the entire dataset once
            avg_loss, per_image_iou = self.train_one_epoch(epoch)

            # Log the statistics
            print(f"Train Loss/IOU: {avg_loss}/{per_image_iou}")
            self.writer.add_scalars("Average Training Loss/IOU",
                {
                    "Loss": avg_loss,
                    "IOU": per_image_iou
                },
                epoch
            )
                
            self.writer.flush()
            
        self.save_model()

    def save_model(self):
        model_path = f"model_checkpoint/model_{self.timestamp}.pt"
        torch.save(self.model.state_dict(), model_path)

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Tomato Leaf Training Program",
        description="Trains a basic U-Net on the tomato leaf data"
    )

    parser.add_argument('-e', '--epoch', default=10, type=int)
    parser.add_argument('-t', '--threshold', default=0.7, type=float)
    parser.add_argument('-d', '--device', default="cpu", type=str)
    parser.add_argument('-n', '--num-workers', default=os.cpu_count() // 2, type=int)

    args = parser.parse_args()

    train_session = TrainSession(args)
    train_session.train(args.epoch)
