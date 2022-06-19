import os

import cv2
import torch
import argparse
import numpy as np
import albumentations as A
import torchmetrics
from albumentations.pytorch import ToTensorV2
from torchvision import models

from core.mixedmodel_cd_based import DependentResUnetMultiDecoder
from core.res_unet import ResUnet
from dataset.s2looking_allmask import S2LookingAllMask
from utils import metrics
from dataset.s2looking import *
from dataset.utils import *
from PIL import Image, ImagePalette
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.hparams import HParam
from utils.images import *
from utils.hungarian import *
from matplotlib import pyplot as plt

def main(hp, mode, weights, trained_path, saved_path, threshold=0.5, batch_size=8, save_sub_mask=False, cd=True):
    assert (0 <= mode < 3)
    assert os.path.isfile(trained_path)

    img1_save_path = os.path.join(saved_path, "img1")
    img2_save_path = os.path.join(saved_path, "img2")
    cd_save_path = os.path.join(saved_path, "cd")
    hungarian_cd_save_path = os.path.join(saved_path, "hungarian_cd")
    # masks_save_path = os.path.join(saved_path, "masks")

    training_metrics = torchmetrics.MetricCollection(
        {
            "Dice": torchmetrics.Dice(average='none', num_classes=2),
        },
        prefix='test_'
    )

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    if not os.path.exists(img1_save_path):
        os.makedirs(img1_save_path)

    if not os.path.exists(img2_save_path):
        os.makedirs(img2_save_path)

    if not os.path.exists(cd_save_path):
        os.makedirs(cd_save_path)

    # if not os.path.exists(masks_save_path):
    #     os.makedirs(masks_save_path)

    resnet = models.resnet34(pretrained=True)
    model = DependentResUnetMultiDecoder(resnet=resnet).cuda()
    checkpoint = torch.load(trained_path)
    state_dict = model.on_load_checkpoint(checkpoint)
    model.load_state_dict(state_dict, strict=False)
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
        additional_targets={'image0': 'image'}
    )

    dataset = S2LookingAllMask(hp.cd_dset_dir, "test", transform, 2, without_mask=False)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=2, shuffle=False
    )

    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()

    criterion = metrics.BCEDiceLoss(weight=[0.1, 0.9])
    hungarian_branch_acc = metrics.MetricTracker()

    loader = tqdm(dataloader, desc="Evaluating")

    NUCLEI_PALETTE = ImagePalette.random()

    with torch.no_grad():
        for (idx, data) in enumerate(loader):
            input1 = data['x'][:, 0, ...].cuda()
            input2 = data['x'][:, 1, ...].cuda()

            output1 = (model(input1).cpu().numpy() >= threshold) * 1
            output2 = (model(input2).cpu().numpy() >= threshold) * 1
            for i in range(output1.shape[0]):
                files = dataset.files[idx * batch_size + i]
                filename = os.path.basename(files['image1'])[:-4]

                masks1 = save_mask_and_contour(output1[i, 0, ...], output1[i, 1, ...], NUCLEI_PALETTE, os.path.join(img1_save_path, "is_{filename}.png".format(filename=filename)), (1024, 1024)).astype(int)
                masks2 = save_mask_and_contour(output2[i, 0, ...], output2[i, 1, ...], NUCLEI_PALETTE, os.path.join(img2_save_path, "is_{filename}.png".format(filename=filename)), (1024, 1024)).astype(int)
             
                if cd:
                    masks1 = torch.from_numpy(masks1)
                    masks2 = torch.from_numpy(masks2)

                    hg_map = change_detection_map(masks1, masks2, dataset.height, dataset.width)
                    gt_cd = (np.array(Image.open(files["mask"])) / 255.0).astype('int')

                    training_metrics(
                        target=torch.from_numpy(gt_cd),
                        preds=torch.from_numpy(hg_map)
                    )
                    hungarian_branch_acc.update(
                        metrics.np_dice_coeff(hg_map[np.newaxis, :, :], gt_cd[np.newaxis, :, :]), 1)

                    hg_img = (hg_map * 255).astype(np.uint8)

                    # mask_color_1 = convert_to_color_map(masks1, dataset.width, dataset.height)
                    # mask_color_2 = convert_to_color_map(masks2, dataset.width, dataset.height)
                    cv2.imwrite(os.path.join(hungarian_cd_save_path, "{filename}.png".format(filename=filename)),
                                hg_img)

    values = training_metrics.compute()
    print(values)
    print("Validation Loss: {:.4f} Acc: {:.4f}".format(valid_loss.avg, hungarian_branch_acc.avg))


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

