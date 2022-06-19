import warnings
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from utils.hparams import HParam
from dataset.s2looking_allmask import S2LookingAllMask
from utils import metrics
from torch.utils.tensorboard import SummaryWriter
from core.mixedmodel_cd_based import DependentResUnetMultiDecoder
import torchmetrics
import torch
import argparse
import os
import ssl

from utils.metrics import TrackingMetric

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


def main(hpconfig, num_epochs, resume, segmentation_weights, name, device, training_weight=None):
    ssl._create_default_https_context = ssl._create_unverified_context

    checkpoint_dir = "{}/{}".format(hpconfig.checkpoints, name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    os.makedirs("{}/{}".format(hpconfig.log, name), exist_ok=True)
    writer = SummaryWriter('runs/{name}'.format(name=name))

    # Model
    resnet = models.resnet50(pretrained=True)
    model = DependentResUnetMultiDecoder(resnet=resnet).to(device)
    # model.change_segmentation_branch_trainable(False)

    # set up binary cross entropy and dice loss
    criterion = metrics.BCEDiceLoss(weight=[0.1, 0.9])

    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.get_siamese_parameter()},
        {'params': model.get_segmentation_parameter(), 'lr': 1e-6},
        {'params': model.get_encoder_parameter(), 'lr': 1e-6},
    ], lr=1e-3)

    # decay LR
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    training_metrics = torchmetrics.MetricCollection(
        {
            "Loss": TrackingMetric(name="loss"),
            "Dice": torchmetrics.Dice(),
            "F1Score": torchmetrics.F1Score()
        },
        prefix='train_'
    )

    validation_metrics = training_metrics.clone(prefix="validation_")

    step = 0
    # starting params
    # s_best_loss = 999
    cd_best_loss = 999
    start_epoch = 0
    # optionally resume from a checkpoint
    if segmentation_weights:
        if os.path.isfile(segmentation_weights):
            print("=> loading segmentation weights '{}'".format(segmentation_weights))
            checkpoint = torch.load(segmentation_weights, map_location=device)
            model.load_segmentation_weight(checkpoint['state_dict'])

    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume, map_location=device)

            start_epoch = checkpoint["epoch"]
            cd_best_loss = checkpoint["cd_best_loss"]
            step = checkpoint["step"]

            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    # get data
    # cd
    cd_dataset_train = S2LookingAllMask(hpconfig.cd_dset_dir, "train_cr")
    cd_dataset_val = S2LookingAllMask(hpconfig.cd_dset_dir, "val")
    cd_dataset_test = S2LookingAllMask(hpconfig.cd_dset_dir, "test")

    cd_train_dataloader = DataLoader(
        cd_dataset_train, batch_size=hpconfig.batch_size, num_workers=2, shuffle=True
    )
    cd_val_dataloader = DataLoader(
        cd_dataset_val, batch_size=hpconfig.batch_size, num_workers=2, shuffle=False
    )
    cd_test_dataloader = DataLoader(
        cd_dataset_test, batch_size=hpconfig.batch_size, num_workers=2, shuffle=False
    )

    for epoch in range(start_epoch, num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # step the learning rate scheduler
        lr_scheduler.step()

        # run training and validation
        # logging accuracy and loss
        s_train_acc = metrics.MetricTracker()
        s_train_loss = metrics.MetricTracker()
        cd_train_acc = metrics.MetricTracker()
        cd_train_loss = metrics.MetricTracker()

        loader = tqdm(cd_train_dataloader, desc="Training")
        # iterate over data
        for data in loader:
            # zero the parameter gradients
            optimizer.zero_grad()

            cd_i1 = data['x'].to(device)
            cd_i2 = data['y'].to(device)
            cd_labels = data['mask'].to(device)

            outputs = model(cd_i1, cd_i2)

            # CD loss
            cd_loss = criterion(outputs['cm'], cd_labels)
            loss = cd_loss

            training_metrics(
                preds=outputs['cm'].cpu(),
                target=cd_labels.type(torch.IntTensor).cpu(),
                value={
                    "loss": loss.cpu()
                }
            )
            # backward
            loss.backward()
            optimizer.step()

            # tensorboard logging
            if (step + 1) % hpconfig.logging_step == 0:
                values = training_metrics.compute()
                for v_key in values:
                    writer.add_scalar(v_key, values[v_key], step)

                loader.set_description(
                    "CD training Loss: {:.4f} dice: {:.4f} f1: {:.4f}".format(
                        values['train_Loss'], values['train_Dice'], values['train_F1Score']
                    )
                )

            # Validation
            if (step + 1) % hpconfig.validation_interval == 0:
                valid_values = validation(
                    cd_val_dataloader, model, criterion, device, training_weight, validation_metrics, True, writer, step
                )
                for v_key in valid_values:
                    writer.add_scalar(v_key, valid_values[v_key], step)

                save_path = os.path.join(
                    checkpoint_dir, "%s_checkpoint_%04d.pt" % (name, step)
                )
                # store best loss and save a model checkpoint
                cd_best_loss = min(valid_values["validation_Loss"], cd_best_loss)
                torch.save(
                    {
                        "step": step,
                        "epoch": epoch,
                        "arch": "ResUnet",
                        "state_dict": model.state_dict(),
                        # "s_best_loss": s_best_loss,
                        "cd_best_loss": cd_best_loss,
                        "optimizer": optimizer.state_dict(),
                    },
                    save_path,
                )
                print("Saved checkpoint to: %s" % save_path)

            step += 1

        print("Test...")
        test_metrics = validation(
            cd_test_dataloader, model, criterion, device, training_weight, validation_metrics
        )
        print(test_metrics)


def validation(
        cd_valid_loader, model, criterion, device, training_weight,
        validation_metrics: torchmetrics.MetricCollection,
        write_log=False, logger=None, step=None
):
    print("\nValidation...")

    # switch to evaluate mode
    model.eval()

    # Iterate over data.
    for data in cd_valid_loader:
        # get the inputs and wrap in Variable
        i1 = data['x'].to(device)
        i2 = data['y'].to(device)
        cd_labels = data['mask'].to(device)

        outputs = model(i1, i2)
        cd_loss = criterion(outputs['cm'], cd_labels)

        validation_metrics(
            #dice_preds=outputs['cm'].cpu(),
            preds=outputs['cm'].cpu(),
            #dice_target=cd_labels.type(torch.IntTensor).cpu(),
            target=cd_labels.type(torch.IntTensor).cpu(),
            value={
                "loss": cd_loss.cpu()
            }
        )

    values = validation_metrics.compute()
    print("Validation: CD Loss: {:.4f} dice: {:.4f} f1: {:.4f}".format(
        values['validation_Loss'], values['validation_Dice'], values['validation_F1Score']
    ))

    model.train()
    validation_metrics.reset()
    return values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Road and Building Extraction")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="yaml file for configuration"
    )
    parser.add_argument(
        "--epochs",
        default=75,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--segmentationweights", default="", type=str, metavar="PATH")
    parser.add_argument("--name", default="default", type=str, help="Experiment name")
    parser.add_argument("--mode", default=0, type=int, help="Training mode")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device ID")
    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, "r") as f:
        hp_str = "".join(f.readlines())

    weights = [1.0, 0.1]
    device = torch.device(args.device)

    main(hp, num_epochs=args.epochs, resume=args.resume, segmentation_weights=args.segmentationweights, name=args.name,
         device=device, training_weight=weights)
