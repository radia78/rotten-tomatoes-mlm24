from utils.training import BaseTrainingSession, BaseTrainingSessionConfig
from utils.data import TomatoLeafDataset, transforms_dict
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from models.configs import DexnedSegmenterConfig
from models.dexinet import DexinedSegmenter
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch.optim as optim
import os

def main(args):
    dataset = TomatoLeafDataset(
        root="data/leaf_veins", 
        transforms=transforms_dict
    )
    model_config = DexnedSegmenterConfig()
    model = DexinedSegmenter(
        classes=model_config.classes,
        activation=model_config.activation,
        pretrained=model_config.pretrained
    )
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    training_args = BaseTrainingSessionConfig(
        dataloader=DataLoader(
            dataset=dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            pin_memory=False
            ),
        model=model,
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
    parser = ArgumentParser(
        prog="Tomato Leaf Training Program",
        description="Trains a basic U-Net on the tomato leaf data"
    )

    parser.add_argument('-e', '--epochs', default=10, type=int)
    parser.add_argument('-d', '--device', default="cpu", type=str)
    parser.add_argument('-n', '--num-workers', default=os.cpu_count() // 2, type=int)
    parser.add_argument('-g', '--grad-accumulation', default=False, type=bool)
    parser.add_argument('-s', '--grad-acc-steps', default=4, type=int)
    parser.add_argument('-sg', '--save-grad', default=False, type=bool)
    parser.add_argument('-b', '--batch-size', default=4, type=int)

    args = parser.parse_args()
    main(args)
