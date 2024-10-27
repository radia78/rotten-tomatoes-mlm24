from utils.training import BaseTrainingSession, BaseTrainingSessionConfig
from utils.data import TomatoLeafDataset, transforms_dict
from torch.utils.data import DataLoader
import models
from models import load_model
from argparse import ArgumentParser
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torch.optim as optim
import os

def main():
    # Parse the system arguments
    parser = ArgumentParser(
        prog="Tomato Leaf Training Program",
        description="Train a selected model for leaf-vein segmentation"
    )
    # Main arguments
    parser.add_argument('model', default="unet", type=str)

    # Training arguments
    parser.add_argument('-e', '--epochs', default=100, type=int)
    parser.add_argument('-t', '--threshold', default=0.7, type=float)
    parser.add_argument('-d', '--device', default="cpu", type=str)
    parser.add_argument('-n', '--num-workers', default=os.cpu_count() // 2, type=int)
    parser.add_argument('-g', '--grad-accumulation', default=False, type=bool)
    parser.add_argument('-s', '--grad-acc-steps', default=4, type=int)
    parser.add_argument('-sg', '--save-grad', default=False, type=bool)
    parser.add_argument('-b', '--batch-size', default=4, type=int)

    args = parser.parse_args()

    # Load the dataset
    dataset = TomatoLeafDataset(
        root="data/leaf_veins", 
        transforms=transforms_dict
    )

    # Load the model and associated configurations
    model_name = args.model
    model, model_config = load_model(model_name)

    # Training components loaded
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    training_args = BaseTrainingSessionConfig(
        dataloader=DataLoader(
            dataset=dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            pin_memory=False
            ),
        model=model,
        model_name=model_name,
        loss_fn=smp.losses.JaccardLoss(
            mode="binary",
            from_logits=True
        ),
        optimizer=optimizer,
        scheduler=optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs*len(dataset),
            eta_min=1e-5
        ),
        device=args.device,
        grad_accumulation=args.grad_accumulation,
        grad_acc_steps=args.grad_acc_steps,
        save_grad=args.save_grad,
    )

    train_session = BaseTrainingSession(training_args)
    train_session.train(args.epochs)

if __name__ == "__main__":
    main()