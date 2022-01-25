import warnings

warnings.simplefilter("ignore", (UserWarning, FutureWarning))
from utils.hparams import HParam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import dataloader
from dataset.crowdaisegment import MappingChallengeDataset
from utils import metrics
from core.res_unet import ResUnet
from core.res_unet_plus import ResUnetPlusPlus
from utils.logger import MyWriter
import torch
import argparse
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

def main(hp, num_epochs, resume, name, training_mode = 0, training_weight = None):

    assert (0 <= training_mode < 3)
    assert (training_mode == 0 or training_weight is not None)

    checkpoint_dir = "{}/{}".format(hp.checkpoints, name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    os.makedirs("{}/{}".format(hp.log, name), exist_ok=True)
    writer = MyWriter("{}/{}".format(hp.log, name))
    # get model

    if hp.RESNET_PLUS_PLUS:
        model = ResUnetPlusPlus(3).cuda()
    else:
        model = ResUnet(3, training_mode + 1).cuda()

    # set up binary cross entropy and dice loss
    criterion = metrics.BCEDiceLoss(weight=[0.1, 0.9])

    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)

    # decay LR
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # starting params
    best_loss = 999
    start_epoch = 0
    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)

            start_epoch = checkpoint["epoch"]

            best_loss = checkpoint["best_loss"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # transform
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.RandomGamma(),
        A.RGBShift(p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    )

    test_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    )

    # get data
    dataset_train = MappingChallengeDataset(hp.dset_dir, "train", training_mode, hp.train_size, train_transform)
    dataset_valid = MappingChallengeDataset(hp.dset_dir, "val", training_mode, hp.test_size, test_transform)

    dataset_train.rand()
    dataset_valid.rand()
    print(len(dataset_train))
    print(len(dataset_valid))
    print(hp.batch_size)
    # creating loaders
    train_dataloader = DataLoader(
        dataset_train, batch_size=hp.batch_size, num_workers=2, shuffle=True
    )
    val_dataloader = DataLoader(
        dataset_valid, batch_size=1, num_workers=2, shuffle=False
    )

    step = 0
    for epoch in range(start_epoch, num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # step the learning rate scheduler
        lr_scheduler.step()

        # run training and validation
        # logging accuracy and loss
        train_acc = metrics.MetricTracker()
        train_loss = metrics.MetricTracker()

        dataset_train.rand()
        dataset_valid.rand()

        # iterate over data

        loader = tqdm(train_dataloader, desc="training")
        for idx, data in enumerate(loader):
            # get the inputs and wrap in Variable
            inputs = data["image"].float().cuda()
            labels = data["mask"].float().cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # prob_map = model(inputs) # last activation was a sigmoid
            # outputs = (prob_map > 0.3).float()
            outputs = model(inputs)
            # outputs = torch.nn.functional.sigmoid(outputs)

            loss = 0.0
            for i in range(training_mode + 1):
                _loss = criterion(outputs[:, i, ...], labels[:, i, ...])
                loss += training_weight[i] * loss

            # backward
            loss.backward()
            optimizer.step()

            train_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
            train_loss.update(loss.data.item(), outputs.size(0))

            # tensorboard logging
            if (step + 1) % hp.logging_step == 0:
                writer.log_training(train_loss.avg, train_acc.avg, step)
                loader.set_description(
                    "Training Loss: {:.4f} Acc: {:.4f}".format(
                        train_loss.avg, train_acc.avg
                    )
                )

            # Validation
            if (step + 1) % hp.validation_interval == 0:
                valid_metrics = validation(
                    val_dataloader, model, criterion, writer, step
                )
                save_path = os.path.join(
                    checkpoint_dir, "%s_checkpoint_%04d.pt" % (name, step)
                )
                # store best loss and save a model checkpoint
                best_loss = min(valid_metrics["valid_loss"], best_loss)
                torch.save(
                    {
                        "step": step,
                        "epoch": epoch,
                        "arch": "ResUnet",
                        "state_dict": model.state_dict(),
                        "best_loss": best_loss,
                        "optimizer": optimizer.state_dict(),
                    },
                    save_path,
                )
                print("Saved checkpoint to: %s" % save_path)

            step += 1

        dataset_train.rand()


def validation(valid_loader, model, criterion, logger, step):

    # logging accuracy and loss
    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()

    # switch to evaluate mode
    model.eval()

    # Iterate over data.
    for idx, data in enumerate(tqdm(valid_loader, desc="validation")):

        # get the inputs and wrap in Variable
        inputs = data["image"].float().cuda()
        labels = data["mask"].cuda()

        # forward
        # prob_map = model(inputs) # last activation was a sigmoid
        # outputs = (prob_map > 0.3).float()
        outputs = model(inputs)
        # outputs = torch.nn.functional.sigmoid(outputs)

        loss = criterion(outputs, labels)

        valid_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
        valid_loss.update(loss.data.item(), outputs.size(0))
        if idx == 0:
            logger.log_images(inputs.cpu(), labels.cpu(), outputs.cpu(), step)
    logger.log_validation(valid_loss.avg, valid_acc.avg, step)

    print("Validation Loss: {:.4f} Acc: {:.4f}".format(valid_loss.avg, valid_acc.avg))
    model.train()
    return {"valid_loss": valid_loss.avg, "valid_acc": valid_acc.avg}


if __name__ == "__main__":
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

    weights = [1.0]
    if args.mode == 1:
        weights = [1.0, 0.1]
    elif args.mode == 2:
        weights = [1.0, 0.1, 0.05]

    main(hp, num_epochs=args.epochs, resume=args.resume, name=args.name, training_mode=args.mode, training_weight=weights)
