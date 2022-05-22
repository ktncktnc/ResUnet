import warnings

import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from utils.hparams import HParam
import numpy as np
from dataset.alabama_segment import AlabamaDataset
from dataset.s2looking_allmask import S2LookingAllMask
from utils import metrics
from core.mixedmodel_cd_based import DependentResUnetMultiDecoder
from utils.logger import MyWriter
from utils.metrics import Dice, TrackingMetric
import torch
import argparse
import os
import ssl

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


def main(hpconfig, num_epochs, resume, name, device, training_weight=None):
    ssl._create_default_https_context = ssl._create_unverified_context

    writer = SummaryWriter('runs/{name}'.format(name=name))

    domain_loss_weight = 0.05
    max_domain_loss_weight = 0.5
    domain_loss_weight_decay = 1.07

    checkpoint_dir = "{}/{}".format(hpconfig.checkpoints, name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    os.makedirs("{}/{}".format(hpconfig.log, name), exist_ok=True)
    # writer = MyWriter("{}/{}".format(hpconfig.log, name))

    # Model
    resnet = models.resnet50(pretrained=True)
    model = DependentResUnetMultiDecoder(resnet=resnet).to(device)

    # set up binary cross entropy and dice loss
    criterion = metrics.BCEDiceLoss(weight=[0.1, 0.9])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hpconfig.lr)

    # decay LR
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    training_metrics = torchmetrics.MetricCollection(
        {
            "Loss": TrackingMetric(name="loss"),
            "Segmentation_Loss": TrackingMetric(name="segmentation_loss"),
            "Domain_Loss": TrackingMetric(name="domain_loss"),
            "Segmentation_Dice": Dice(),
            "Domain_Accuracy": torchmetrics.classification.Accuracy(),
        },
        prefix='train_'
    )
    validation_metrics = training_metrics.clone(prefix="validation_")

    step = 0
    # starting params
    s_best_loss = 999
    start_epoch = 0
    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume, map_location=device)

            start_epoch = checkpoint["epoch"]
            s_best_loss = checkpoint["s_best_loss"]
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
    alb_dataset_train = AlabamaDataset(hpconfig.dset1_dir, "train")
    alb_dataset_val = AlabamaDataset(hpconfig.dset1_dir, "test")
    s2l_dataset_train = S2LookingAllMask(hpconfig.dset2_dir, "train")
    s2l_dataset_val = S2LookingAllMask(hpconfig.dset2_dir, "test")

    alb_train_dataloader = DataLoader(
        alb_dataset_train, batch_size=hpconfig.batch_size, num_workers=2, shuffle=True
    )
    alb_val_dataloader = DataLoader(
        alb_dataset_val, batch_size=hpconfig.batch_size, num_workers=2, shuffle=False
    )

    s2l_train_dataloader = DataLoader(
        s2l_dataset_train, batch_size=int(hpconfig.batch_size / 2), num_workers=2, shuffle=True
    )
    s2l_val_dataloader = DataLoader(
        s2l_dataset_val, batch_size=int(hpconfig.batch_size / 2), num_workers=2, shuffle=False
    )

    loader_len = min(len(alb_dataset_train), len(s2l_train_dataloader))
    nllloss = torch.nn.NLLLoss().to(device)

    for epoch in range(start_epoch, num_epochs):
        alb_train_batch = iter(alb_train_dataloader)
        s2l_train_batch = iter(s2l_train_dataloader)

        start_steps = epoch * loader_len
        total_steps = num_epochs * loader_len

        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # step the learning rate scheduler
        lr_scheduler.step()

        loader = tqdm(range(loader_len), desc="Training")
        # iterate over data
        for i in loader:
            # zero the parameter gradients
            optimizer.zero_grad()

            p = float(i + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            s_data = next(alb_train_batch)
            s_images = s_data['image'].to(device)
            s_groundtruth = s_data['mask'].to(device)
            s_domains = torch.zeros(s_images.shape[0]).long().to(device)

            t_data = next(s2l_train_batch)
            t_image1 = t_data['x'].to(device)
            t_image2 = t_data['y'].to(device)
            t_domains = torch.ones(t_image1.shape[0] + t_image2.shape[0]).long().to(device)

            s_outputs, s_output_domains = model.segment_forward(s_images, alpha=alpha)
            t_output_domains1 = model.domain_classify(x=t_image1, alpha=alpha)
            t_output_domains2 = model.domain_classify(x=t_image2, alpha=alpha)
            t_output_domains = torch.cat((t_output_domains1, t_output_domains2), 0)

            domains = torch.cat((s_domains, t_domains), dim=0).cpu()
            output_domains = torch.cat((s_output_domains, t_output_domains), 0).cpu()

            # Segment loss
            segment_loss = 0.0
            for i in range(2):
                _loss = criterion(s_outputs[:, i, ...], s_groundtruth[:, i, ...])
                segment_loss += training_weight[i] * _loss

            domain_loss = nllloss(s_output_domains, s_domains) + nllloss(t_output_domains, t_domains)
            loss = segment_loss + domain_loss

            training_metrics(
                preds=output_domains.cpu(),
                target=domains.cpu(),
                dice_preds=s_outputs[:, 0, ...].cpu(),
                dice_target=s_groundtruth[:, 0, ...].type(torch.IntTensor).cpu(),
                value={
                    "loss": loss.cpu(),
                    "segmentation_loss": segment_loss.cpu(),
                    "domain_loss": domain_loss.cpu()
                }
            )

            # cd_train_acc.update(metrics.dice_coeff(outputs['cm'], cd_labels), outputs['cm'].size(0))
            # cd_train_loss.update(cd_loss.data.item(), outputs['cm'].size(0))

            # s_train_acc.update(metrics.dice_coeff(s_outputs[:, 0, ...], s_groundtruth[:, 0, ...]), s_outputs.size(0))
            # s_train_loss.update(loss.data.item(), s_outputs.size(0))
            # s_domain_acc.update((metrics.acc(s_output_domains, s_domains) + metrics.acc(t_output_domains, t_domains))/2.0, 1)
            # backward
            loss.backward()
            optimizer.step()

            # tensorboard logging
            if (step + 1) % hpconfig.logging_step == 0:
                values = training_metrics.compute()
                loader.set_description(
                    "Training: loss: {:.4f} domain loss: {:.4f} dice: {:.4f}".format(
                        values["train_Loss"], values["train_Domain_Loss"], values["train_Segmentation_Dice"]
                    )
                )
                writer.add_scalar("alpha", alpha, step)
                writer.add_scalar("train_Loss", values["train_Loss"], step)
                writer.add_scalar("train_Segmentation_Loss", values["train_Segmentation_Loss"], step)
                writer.add_scalar("train_Domain_Loss", values["train_Domain_Loss"], step)
                writer.add_scalar("train_Segmentation_Dice", values["train_Segmentation_Dice"], step)
                writer.add_scalar("train_Domain_Accuracy", values["train_Domain_Accuracy"], step)

            # Validation
            if (step + 1) % hpconfig.validation_interval == 0:
                valid_metrics = validation(
                    alb_val_dataloader, s2l_val_dataloader, model, criterion, device, training_weight,
                    domain_loss_weight, alpha, validation_metrics
                )

                writer.add_scalar("validation_Loss", valid_metrics["validation_Loss"], step)
                writer.add_scalar("validation_Segmentation_Loss", valid_metrics["validation_Segmentation_Loss"], step)
                writer.add_scalar("validation_Domain_Loss", valid_metrics["validation_Domain_Loss"], step)
                writer.add_scalar("validation_Segmentation_Dice", valid_metrics["validation_Segmentation_Dice"], step)
                writer.add_scalar("validation_Domain_Accuracy", valid_metrics["validation_Domain_Accuracy"], step)

                save_path = os.path.join(
                    checkpoint_dir, "%s_checkpoint_%04d.pt" % (name, step)
                )
                # store best loss and save a model checkpoint
                s_best_loss = min(valid_metrics["validation_Loss"], s_best_loss)
                torch.save(
                    {
                        "step": step,
                        "epoch": epoch,
                        "arch": "ResUnet",
                        "state_dict": model.state_dict(),
                        "s_best_loss": s_best_loss,
                        "optimizer": optimizer.state_dict(),
                    },
                    save_path,
                )
                print("Saved checkpoint to: %s" % save_path)

            step += 1

        domain_loss_weight = min(max_domain_loss_weight, domain_loss_weight * domain_loss_weight_decay)
        training_metrics.reset()


def validation(s_dataloader, t_dataloader, model, criterion, device, training_weight, domain_loss_weight, alpha,
               validation_metrics: torchmetrics.MetricCollection):
    print("\nValidation...")

    s_batch = iter(s_dataloader)
    t_batch = iter(t_dataloader)

    loader_len = min(len(s_dataloader), len(t_dataloader))
    nllloss = torch.nn.NLLLoss().to(device)

    # switch to evaluate mode
    model.eval()

    # Iterate over data.
    for i in range(5):
        # get the inputs and wrap in Variable
        s_data = next(s_batch)
        s_images = s_data['image'].to(device)
        s_groundtruth = s_data['mask'].to(device)
        s_domains = torch.zeros(s_images.shape[0]).long().to(device)

        t_data = next(t_batch)
        t_image1 = t_data['x'].to(device)
        t_image2 = t_data['y'].to(device)
        t_domains = torch.ones(t_image1.shape[0] + t_image2.shape[0]).long().to(device)
        domains = torch.cat((s_domains, t_domains), 0)

        s_outputs, s_output_domains = model.segment_forward(s_images, alpha=alpha)
        t_output_domains1 = model.domain_classify(x=t_image1, alpha=alpha)
        t_output_domains2 = model.domain_classify(x=t_image2, alpha=alpha)
        t_output_domains = torch.cat((t_output_domains1, t_output_domains2), 0)
        output_domains = torch.cat((s_output_domains, t_output_domains), 0)

        segment_loss = 0.0
        for i in range(2):
            _loss = criterion(s_outputs[:, i, ...], s_groundtruth[:, i, ...])
            segment_loss += training_weight[i] * _loss

        domain_loss = nllloss(s_output_domains, s_domains) + nllloss(t_output_domains, t_domains)
        loss = segment_loss + domain_loss

        validation_metrics(
            preds=output_domains.cpu(),
            target=domains.cpu(),
            dice_preds=s_outputs[:, 0, ...].cpu(),
            dice_target=s_groundtruth[:, 0, ...].type(torch.IntTensor).cpu(),
            value={
                "loss": loss.cpu(),
                "segmentation_loss": segment_loss.cpu(),
                "domain_loss": domain_loss.cpu()
            }
        )

    values = validation_metrics.compute()
    print(
        "Validation: loss: {:.4f} domain loss: {:.4f} dice: {:.4f}".format(
            values["validation_Loss"], values["validation_Domain_Loss"], values["validation_Segmentation_Dice"]
        )
    )

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
    parser.add_argument("--name", default="default", type=str, help="Experiment name")
    parser.add_argument("--mode", default=0, type=int, help="Training mode")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device ID")
    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, "r") as f:
        hp_str = "".join(f.readlines())

    weights = [1.0, 0.1]
    device = torch.device(args.device)

    main(hp, num_epochs=args.epochs, resume=args.resume, name=args.name, device=device, training_weight=weights)
