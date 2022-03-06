import warnings
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from utils.hparams import HParam
from dataset.crowdaisegment import MappingChallengeDataset
from dataset.s2looking import S2Looking
from utils import metrics
from core.mixedmodel import ResUnetMultiDecoder
from utils.logger import MyWriter
import torch
import argparse
import os
import albumentations as albums
from albumentations.pytorch import ToTensorV2
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


def main(hp, num_epochs, resume, name, training_weight=None):
    checkpoint_dir = "{}/{}".format(hp.checkpoints, name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    os.makedirs("{}/{}".format(hp.log, name), exist_ok=True)
    writer = MyWriter("{}/{}".format(hp.log, name))

    # Model
    resnet = models.resnet34(pretrained=True)
    model = ResUnetMultiDecoder(resnet=resnet).cuda()

    # set up binary cross entropy and dice loss
    criterion = metrics.BCEDiceLoss(weight=[0.1, 0.9])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)

    # decay LR
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

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
            'border_mask': 'mask',
            'touching_mask': 'mask'
        }
    )

    test_transform = albums.Compose([
        albums.Resize(256, 256),
        albums.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ],
        additional_targets={
            'border_mask': 'mask',
            'touching_mask': 'mask'
        }
    )

    # get data
    # segment
    s_dataset_train = MappingChallengeDataset(hp.segment_dset_dir, "train", 1, hp.segment_train_size,
                                              train_transform)
    s_dataset_valid = MappingChallengeDataset(hp.segment_dset_dir, "val", 1, hp.segment_test_size, test_transform)

    s_dataset_train.rand()
    s_dataset_valid.rand()

    # cd
    cd_dataset_train = S2Looking(hp.cd_dset_dir, "train", train_transform)
    cd_dataset_test = S2Looking(hp.cd_dset_dir, "val", test_transform)

    # creating loaders
    s_train_dataloader = DataLoader(
        s_dataset_train, batch_size=hp.batch_size, num_workers=2, shuffle=True
    )
    s_val_dataloader = DataLoader(
        s_dataset_valid, batch_size=hp.batch_size, num_workers=2, shuffle=False
    )

    cd_train_dataloader = DataLoader(
        cd_dataset_train, batch_size=hp.batch_size, num_workers=2, shuffle=True
    )
    cd_val_dataloader = DataLoader(
        cd_dataset_test, batch_size=hp.batch_size, num_workers=2, shuffle=False
    )

    s_step = cd_step = 0
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

        s_dataset_train.rand()

        # iterate over data
        # segment
        loader = tqdm(s_train_dataloader, desc="segment training")
        for idx, data in enumerate(loader):
            # get the inputs and wrap in Variable
            inputs = data["image"].float().cuda()
            labels = data["mask"].float().cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = 0.0
            for i in range(2):
                _loss = criterion(outputs[:, i, ...], labels[:, i, ...])
                loss += training_weight[i] * _loss

            # backward
            loss.backward()
            optimizer.step()

            s_train_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
            s_train_loss.update(loss.data.item(), outputs.size(0))

            # tensorboard logging
            if (s_step + 1) % hp.logging_step == 0:
                writer.log_training(s_train_loss.avg, s_train_acc.avg, s_step, "s_training")
                loader.set_description(
                    "Segment training Loss: {:.4f} Acc: {:.4f}".format(
                        s_train_loss.avg, s_train_acc.avg
                    )
                )

            # Validation
            if (s_step + 1) % hp.validation_interval == 0:
                valid_metrics = validation(
                    s_val_dataloader, model, True, criterion, writer, s_step, training_weight
                )
                save_path = os.path.join(
                    checkpoint_dir, "%s_segment_checkpoint_%04d.pt" % (name, s_step)
                )
                # store best loss and save a model checkpoint
                s_best_loss = min(valid_metrics["valid_loss"], s_best_loss)
                torch.save(
                    {
                        "step": s_step,
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

            s_step += 1

        loader = tqdm(cd_train_dataloader, desc="cd training")
        for idx, data in enumerate(loader):
            i1 = data['x'][:, 0, :, :, :].float().cuda()
            i2 = data['x'][:, 1, :, :, :].float().cuda()
            labels = (data['mask'][:, :, :, 0] / 255.0).float().cuda()
            optimizer.zero_grad()
            outputs = model(i1, i2, True)
            loss = criterion(outputs, labels)
            # backward
            loss.backward()
            optimizer.step()

            cd_train_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
            cd_train_loss.update(loss.data.item(), outputs.size(0))

            # tensorboard logging
            if (cd_step + 1) % hp.logging_step == 0:
                writer.log_training(cd_train_loss.avg, cd_train_acc.avg, cd_step, "cd_training")
                loader.set_description(
                    "CD training Loss: {:.4f} Acc: {:.4f}".format(
                        cd_train_loss.avg, cd_train_acc.avg
                    )
                )

            # Validation
            if (cd_step + 1) % hp.validation_interval == 0:
                valid_metrics = validation(
                    cd_val_dataloader, model, False, criterion, writer, cd_step, training_weight
                )
                save_path = os.path.join(
                    checkpoint_dir, "%s_cd_checkpoint_%04d.pt" % (name, cd_step)
                )
                # store best loss and save a model checkpoint
                cd_best_loss = min(valid_metrics["valid_loss"], cd_best_loss)
                torch.save(
                    {
                        "step": cd_step,
                        "epoch": epoch,
                        "arch": "ResUnet",
                        "state_dict": model.state_dict(),
                        "cd_best_loss": cd_best_loss,
                        "s_best_loss": s_best_loss,
                        "optimizer": optimizer.state_dict(),
                    },
                    save_path,
                )
                print("Saved checkpoint to: %s" % save_path)
            cd_step += 1


def validation(valid_loader, model, is_segment, criterion, logger, step, training_weight):
    print("Validation...")
    # logging accuracy and loss
    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()

    # switch to evaluate mode
    model.eval()

    # Iterate over data.
    for idx, data in enumerate(valid_loader):
        # get the inputs and wrap in Variable
        if is_segment:
            inputs = data["image"].float().cuda()
            labels = data["mask"].cuda()
            # forward
            outputs = model(inputs)
        else:
            i1 = (data['x'][:, 0, :, :, :].float()).cuda()
            i2 = (data['x'][:, 1, :, :, :].float()).cuda()
            labels = (data['mask'][:, :, :, 0] / 255.0).float().cuda()
            outputs = model(i1, i2, True)

        loss = 0.0
        if is_segment:
            n_masks = 2
        else:
            n_masks = 1
        for i in range(n_masks + 1):
            _loss = criterion(outputs[:, i, ...], labels[:, i, ...])
            loss += training_weight[i] * _loss

        valid_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
        valid_loss.update(loss.data.item(), outputs.size(0))
        # if idx == 0:
        #     logger.log_images(inputs.cpu(), labels.cpu(), outputs.cpu(), step)

    logger.log_validation(valid_loss.avg, valid_acc.avg, step)

    valid_text = "Segment validation loss: " if is_segment else "CD validation loss: "
    print(valid_text + "{:.4f} Acc: {:.4f}".format(valid_loss.avg, valid_acc.avg))
    model.train()
    return {"valid_loss": valid_loss.avg, "valid_acc": valid_acc.avg}

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
    parser.add_argument("--mode", default=0, type = int, help = "Training mode")
    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, "r") as f:
        hp_str = "".join(f.readlines())

    weights = [1.0, 0.1]

    main(hp, num_epochs=args.epochs, resume=args.resume, name=args.name, training_weight=weights)
