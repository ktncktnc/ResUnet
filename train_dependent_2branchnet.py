import warnings
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from utils.hparams import HParam
from dataset.s2looking_allmask import S2LookingAllMask
from utils import metrics
from core.mixedmodel_cd_based import DependentResUnetMultiDecoder
from utils.logger import MyWriter
import torch
import argparse
import os
import albumentations as albums
from albumentations.pytorch import ToTensorV2
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


def main(hpconfig, num_epochs, resume, name, training_weight=None):
    checkpoint_dir = "{}/{}".format(hpconfig.checkpoints, name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    os.makedirs("{}/{}".format(hpconfig.log, name), exist_ok=True)
    writer = MyWriter("{}/{}".format(hpconfig.log, name))

    # Model
    resnet = models.resnet34(pretrained=True)
    model = DependentResUnetMultiDecoder(resnet=resnet).cuda()

    # set up binary cross entropy and dice loss
    criterion = metrics.BCEDiceLoss(weight=[0.1, 0.9])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hpconfig.lr)

    # decay LR
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    step = 0
    # starting params
    s_best_loss = 999
    cd_best_loss = 999
    start_epoch = 0
    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)

            start_epoch = checkpoint["epoch"]

            s_best_loss = checkpoint["s_best_loss"]
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

    # transform
    train_transform = albums.Compose([
        albums.Resize(256, 256),
        albums.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.5, 0.1), rotate_limit=10),
        albums.RandomGamma(),
        albums.RGBShift(p=0.2),
        albums.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        albums.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ],
        additional_targets={
            'image0': 'image',
            'mask1': 'mask',
            'mask2': 'mask',
            'border_mask1': 'mask',
            'border_mask2': 'mask'
        }
    )

    test_transform = albums.Compose([
        albums.Resize(256, 256),
        albums.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ],
        additional_targets={
            'image0': 'image',
            'mask1': 'mask',
            'mask2': 'mask',
            'border_mask1': 'mask',
            'border_mask2': 'mask'
        }
    )

    # get data
    # cd
    cd_dataset_train = S2LookingAllMask(hpconfig.cd_dset_dir, "train", train_transform)
    cd_dataset_test = S2LookingAllMask(hpconfig.cd_dset_dir, "val", test_transform)

    cd_train_dataloader = DataLoader(
        cd_dataset_train, batch_size=hpconfig.batch_size, num_workers=2, shuffle=True
    )
    cd_val_dataloader = DataLoader(
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

            cd_i1 = data['x'].cuda()
            cd_i2 = data['y'].cuda()
            cd_labels = (data['mask'][..., 0] / 255.0).cuda()
            cd_labels1 = (data['mask1'] / 255.0).cuda()
            cd_labels2 = (data['mask2'] / 255.0).cuda()

            outputs = model(cd_i1, cd_i2)

            # CD loss
            cd_loss = criterion(outputs['cm'], cd_labels)

            # Segment loss
            segment_loss = 0.0
            for i in range(2):
                _loss1 = criterion(outputs['x'][:, i, ...], cd_labels1[:, i, ...])
                _loss2 = criterion(outputs['y'][:, i, ...], cd_labels2[:, i, ...])
                segment_loss += training_weight[i] * (_loss1 + _loss2)/2.0

            loss = cd_loss + segment_loss*0.9

            cd_train_acc.update(metrics.dice_coeff(outputs['cm'], cd_labels), outputs['cm'].size(0))
            cd_train_loss.update(cd_loss.data.item(), outputs['cm'].size(0))

            s_train_acc.update(metrics.dice_coeff(outputs['x'], cd_labels1), outputs['x'].size(0))
            s_train_loss.update(segment_loss.data.item(), outputs['x'].size(0))
            # backward
            loss.backward()
            optimizer.step()

            # tensorboard logging
            if (step + 1) % hpconfig.logging_step == 0:
                writer.log_training(s_train_loss.avg, s_train_acc.avg, step, "s_training")
                writer.log_training(cd_train_loss.avg, cd_train_acc.avg, step, "cd_training")
                loader.set_description(
                    "S training Loss: {:.4f} acc: {:.4f} CD training Loss: {:.4f} acc: {:.4f}".format(
                        s_train_loss.avg, s_train_acc.avg, cd_train_loss.avg, cd_train_acc.avg
                    )
                )

            # Validation
            if (step + 1) % hpconfig.validation_interval == 0:
                valid_metrics = validation(
                    cd_val_dataloader, model, criterion, writer, step, training_weight
                )
                save_path = os.path.join(
                    checkpoint_dir, "%s_checkpoint_%04d.pt" % (name, step)
                )
                # store best loss and save a model checkpoint
                s_best_loss = min(valid_metrics["s_valid_loss"], s_best_loss)
                cd_best_loss = min(valid_metrics["cd_valid_loss"], cd_best_loss)
                torch.save(
                    {
                        "step": step,
                        "epoch": epoch,
                        "arch": "ResUnet",
                        "state_dict": model.state_dict(),
                        "s_best_loss": s_best_loss,
                        "cd_best_loss": cd_best_loss,
                        "optimizer": optimizer.state_dict(),
                    },
                    save_path,
                )
                print("Saved checkpoint to: %s" % save_path)

            step += 1


def validation(cd_valid_loader, model, criterion, logger, step, training_weight):
    print("\nValidation...")
    # logging accuracy and loss
    s_valid_acc = metrics.MetricTracker()
    s_valid_loss = metrics.MetricTracker()
    cd_valid_acc = metrics.MetricTracker()
    cd_valid_loss = metrics.MetricTracker()

    # switch to evaluate mode
    model.eval()

    # Iterate over data.
    for data in cd_valid_loader:
        # get the inputs and wrap in Variable
        i1 = data['x'].cuda()
        i2 = data['y'].cuda()
        cd_labels = (data['mask'][..., 0] / 255.0).cuda()
        cd_labels1 = (data['mask1'] / 255.0).cuda()
        cd_labels2 = (data['mask2'] / 255.0).cuda()

        outputs = model(i1, i2)
        cd_loss = criterion(outputs['cm'], cd_labels)

        s_loss = 0.0
        for i in range(2):
            _loss1 = criterion(outputs['x'][:, i, ...], cd_labels1[:, i, ...])
            _loss2 = criterion(outputs['y'][:, i, ...], cd_labels2[:, i, ...])
            s_loss += training_weight[i] * _loss1
            s_loss += training_weight[i] * _loss2

        cd_valid_acc.update(metrics.dice_coeff(outputs['cm'], cd_labels), outputs['cm'].size(0))
        cd_valid_loss.update(cd_loss.data.item(), outputs['cm'].size(0))

        s_valid_acc.update(metrics.dice_coeff(outputs['x'], cd_labels1), outputs['x'].size(0))
        s_valid_loss.update(s_loss.data.item(), outputs['x'].size(0))

    logger.log_validation(s_valid_acc.avg, s_valid_loss.avg, step, "s_validation")
    logger.log_validation(cd_valid_acc.avg, cd_valid_loss.avg, step, "cd_validation")

    print("Segment validation loss: {:.4f} Acc: {:.4f} CD validation loss: {:.4f} Acc: {:.4f}"
          .format(s_valid_loss.avg, s_valid_acc.avg, cd_valid_loss.avg, cd_valid_acc.avg))

    model.train()

    return {
        "s_valid_loss": s_valid_loss.avg,
        "s_valid_acc": s_valid_acc.avg,
        "cd_valid_loss": cd_valid_loss.avg,
        "cd_valid_acc": cd_valid_acc.avg
    }


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
    parser.add_argument("--name", default="default", type=str, help="Experiment name")
    parser.add_argument("--mode", default=0, type=int, help="Training mode")
    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, "r") as f:
        hp_str = "".join(f.readlines())

    weights = [1.0, 0.1]

    main(hp, num_epochs=args.epochs, resume=args.resume, name=args.name, training_weight=weights)
