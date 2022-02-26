import os
import torch
import argparse
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from core.res_unet import ResUnet
from utils import metrics
from dataset.crowdaisegment import MappingChallengeDataset
from dataset.utils import *
from PIL import Image, ImagePalette
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.hparams import HParam


def main(hp, mode, weights, trained_path, saved_path, threshold=0.5, batch_size=8, save_sub_mask=False):
    assert (0 <= mode < 3)
    assert os.path.isfile(trained_path)

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    model = ResUnet(3, mode + 1).cuda()
    checkpoint = torch.load(trained_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if save_sub_mask:
        n_masks = mode + 1
    else:
        n_masks = 1

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ],
        additional_targets={
            'border_mask': 'mask',
            'touching_mask': 'mask'
        }
    )

    dataset = MappingChallengeDataset(hp.dset_dir, "val", mode, hp.test_size, transform)
    dataset.rand()

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=2, shuffle=False
    )

    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()

    criterion = metrics.BCEDiceLoss(weight=[0.1, 0.9])

    loader = tqdm(dataloader, desc="Evaluating")

    NUCLEI_PALETTE = ImagePalette.random()

    with torch.no_grad():
        for idx, data in enumerate(loader):
            # get the inputs and wrap in Variable
            inputs = data["image"].float().cuda()
            labels = data["mask"].float().cuda()

            outputs = model(inputs)

            loss = 0.0
            for i in range(mode + 1):
                _loss = criterion(outputs[:, i, ...], labels[:, i, ...])
                loss += weights[i] * _loss

            valid_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
            valid_loss.update(loss.data.item(), outputs.size(0))

            imgs = (outputs.cpu().numpy() >= threshold).astype(np.uint8)

            for i in range(batch_size):
                filename = dataset.get_file_name(idx * batch_size + i)

                # for m in range(n_masks):
                #     Image.fromarray(imgs[i, m, ...]).save(os.path.join(saved_path, "{filename}_{mode}.png".format(filename=filename, mode=m)))

                mask = imgs[i, 0, ...]
                contour = imgs[i, 1, ...]
                seed = ((mask * (1 - contour)) > 0.5).astype(np.uint8)

                labels = label_watershed(mask, seed)
                im = Image.fromarray(labels.astype(np.uint8), mode='P')
                im.putpalette(NUCLEI_PALETTE)
                im.save(os.path.join(saved_path, "is_{filename}.png".format(filename=filename)))

    print("Validation Loss: {:.4f} Acc: {:.4f}".format(valid_loss.avg, valid_acc.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate ResUnet")
    parser.add_argument("--mode", default=1)
    parser.add_argument("--pretrain", type=str)
    parser.add_argument("--savepath", type=str)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--batchsize", default=8, type=int)
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="yaml file for configuration"
    )

    args = parser.parse_args()

    weights = [1.0]
    if int(args.mode) == 1:
        weights = [1.0, 0.1]
    elif int(args.mode) == 2:
        weights = [1.0, 0.1, 0.05]

    hp = HParam(args.config)
    main(hp, int(args.mode), weights, args.pretrain, args.savepath, args.threshold, args.batchsize)

