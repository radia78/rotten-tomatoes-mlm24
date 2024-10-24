import models.configs
import models.dexinet
import models.dexiunet
from utils.training import BaseTrainingSession, BaseTrainingSessionConfig
from utils.data import TomatoLeafDataset, transforms_dict
from torch.utils.data import DataLoader
import models
from argparse import ArgumentParser
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torch.optim as optim
import os

def load_model(model_name: str="unet"):
    model_names_list = [
        "unet",
        "dexined-segmenter",
        "dexiunet"
    ]

    if model_name not in model_names_list:
        raise ModuleNotFoundError(f"No model, {model_name} exists.")
    
    else:
        match model_name:
            case "unet":
                model_config = models.configs.UnetConfig()
                model = smp.Unet(
                    encoder_name=model_config.encoder_name,
                    encoder_weights=model_config.encoder_weights,
                    in_channels=model_config.in_channels,
                    decoder_use_batchnorm=model_config.decoder_use_batchnorm,
                    decoder_attention_type=model_config.decoder_attention_type,
                    classes=model_config.classes
                )
            
            case "dexined-segmenter":
                model_config = models.configs.DexinedSegmenterConfig()
                model = models.dexinet.DexinedSegmenter(
                    classes=model_config.classes,
                    activation=model_config.activation,
                    pretrained=model_config.pretrained
                )

            case "dexiunet":
                model_config = models.configs.DexiUnetConfig()
                model = models.dexiunet.DexiUnet(
                    encoder_channels=model_config.encoder_channels,
                    decoder_channels=model_config.decoder_channels,
                    decoder_depth=model_config.decoder_depth,
                    use_batchnorm=model_config.use_batchnorm,
                    attention_type=model_config.attention_type,
                    center=model_config.center,
                    classes=model_config.classes, 
                    activation=model_config.activation, 
                    pretrained=model_config.pretrained
                )

        return model, model_config

def main():
    # Parse the system arguments
    parser = ArgumentParser(
        prog="Tomato Leaf Training Program",
        description="Train a selected model for leaf-vein segmentation"
    )
    # Main arguments
    parser.add_argument('model', default="unet", type=str)

    # Training arguments
    parser.add_argument('-e', '--epoch', default=100, type=int)
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
    model, model_config = load_model(args.model)

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