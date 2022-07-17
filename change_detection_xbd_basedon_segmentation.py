import os
from torchvision import models

import cv2
import torch
import argparse
import numpy as np
import albumentations as albums
import torchmetrics
from albumentations.pytorch import ToTensorV2
from core.mixedmodel_cd_based import DependentResUnetMultiDecoder
from utils import metrics
from dataset.xbd import XView2Dataset
from dataset.utils import *
from PIL import Image, ImagePalette
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.hparams import HParam
from utils.images import *
from utils.hungarian import *


def main(hp, mode, weights, device, split, trained_path, saved_path, threshold=0.5, batch_size=8, save_sub_mask=False,
         cm_weights=None):
    if cm_weights is None:
        cm_weights = [0.3, 0.7]
    assert (0 <= mode < 3)
    assert os.path.isfile(trained_path)

    img1_save_path = os.path.join(saved_path, "img1")
    img2_save_path = os.path.join(saved_path, "img2")

    cd_save_path = os.path.join(saved_path, "cd")
    hungarian_cd_save_path = os.path.join(saved_path, "hungarian_cd")
    final_cd_path = os.path.join(saved_path, "final_cd")

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    if not os.path.exists(img1_save_path):
        os.makedirs(img1_save_path)

    if not os.path.exists(img2_save_path):
        os.makedirs(img2_save_path)

    if not os.path.exists(cd_save_path):
        os.makedirs(cd_save_path)

    if not os.path.exists(hungarian_cd_save_path):
        os.makedirs(hungarian_cd_save_path)

    if not os.path.exists(final_cd_path):
        os.makedirs(final_cd_path)

    resnet = models.resnet50(pretrained=True)
    model = DependentResUnetMultiDecoder(resnet=resnet, input_channel=4, cd_o_channel=5).to(device)
    checkpoint = torch.load(trained_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    training_metrics = torchmetrics.MetricCollection(
        {
            "Dice": torchmetrics.Dice(average='none', num_classes=5),
        },
        prefix='test_'
    )

    if save_sub_mask:
        n_masks = mode + 1
    else:
        n_masks = 1

    #dataset = S2LookingAllMask(hp.cd_dset_dir, split)
    dataset = XView2Dataset(root_dir=hp.cd_dset_dir, mode=split)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=2, shuffle=False
    )

    cd_branch_acc = metrics.MetricTracker()
    hungarian_branch_acc = metrics.MetricTracker()
    final_acc = metrics.MetricTracker()

    criterion = metrics.BCEDiceLoss(weight=[0.1, 0.9])

    loader = tqdm(dataloader, desc="Evaluating")

    nuclei_palette = ImagePalette.load('/root/img_palette.txt')

    img_height, img_width = dataset.get_full_resized_shape()
    full_cm = np.zeros((5, img_height, img_width))
    full_label = np.zeros((img_height, img_width))
    full_x = np.zeros((2, img_height, img_width))
    full_y = np.zeros((2, img_height, img_width))
    full_x_probs = np.zeros((img_height, img_width), dtype=np.float64)
    full_y_probs = np.zeros((img_height, img_width), dtype=np.float64)
    full_cm_probs = np.zeros((5, img_height, img_width), dtype=np.float64)

    with torch.no_grad():
        for (idx, data) in enumerate(loader):
            cd_i1 = data['x'].to(device)
            cd_i2 = data['y'].to(device)
            cd_labels = data['masks'].to(device)
            outputs = model.siamese_forward(cd_i1, cd_i2)

            #cd_branch_acc.update(metrics.dice_coeff(outputs['cm'], cd_labels), outputs['cm'].size(0))

            cm_probs = outputs.cpu().numpy()
            #x_probs = outputs['x'].cpu().numpy()
            #y_probs = outputs['y'].cpu().numpy()

            cm = (cm_probs >= threshold) * 1
            #x = (x_probs >= threshold) * 1
            #y = (y_probs >= threshold) * 1

            hg_probs = []
            final_probs = []

            for i in range(cm.shape[0]):
                # Get file name
                files = dataset.files[idx * batch_size + i]
                divide = files['divide']
                x1, x2, y1, y2 = dataset.get_resized_coord(divide)
                pre_name = os.path.basename(files['pre_img'])[:-4]
                post_name = os.path.basename(files['post_img'])[:-4]

                #full_x[:, x1:x2, y1:y2] = x[i, :, ...]
                #full_y[:, x1:x2, y1:y2] = y[i, :, ...]
                full_cm[:, x1:x2, y1:y2] = cm[i, :, ...]

                #full_x_probs[x1:x2, y1:y2] = x_probs[i, 0, ...]
                #full_y_probs[x1:x2, y1:y2] = y_probs[i, 0, ...]
                full_cm_probs[:, x1:x2, y1:y2] = cm_probs[i, :, ...]

                if divide >= dataset.divide*dataset.divide - 1:

                    # Colorize instance segmentation map and save
                    # masks1 = save_mask_and_contour(
                    #     full_x[0, ...], full_x[1, ...], nuclei_palette,
                    #     os.path.join(img1_save_path, "mask1_{filename}.png".format(filename=filename)),
                    #     (dataset.width, dataset.height)
                    # ).astype(int)

                    # masks2 = save_mask_and_contour(
                    #     full_y[0, ...], full_y[1, ...], nuclei_palette,
                    #     os.path.join(img2_save_path, "mask2_{filename}.png".format(filename=filename)),
                    #     (dataset.width, dataset.height)
                    # ).astype(int)

                    # masks1 = torch.from_numpy(masks1)
                    # masks2 = torch.from_numpy(masks2)

                    # Hungarian algorithm
                    # hg_map = change_detection_map(masks1, masks2, 256, 256)
                    # hg_img = (hg_map * 255).astype(np.uint8)

                    # mask_color_1 = convert_to_color_map(masks1)
                    # mask_color_2 = convert_to_color_map(masks2)
                    # plot_and_save(mask_color_1, mask_color_2, hg_img,
                    #               os.path.join(hungarian_cd_save_path, "{filename}.png".format(filename=filename)))

                    one_channel_full_cm = np.argmax(full_cm_probs, axis=0)
                    cm_img = Image.fromarray(one_channel_full_cm, mode='P')
                    cm_img = cm_img.resize((dataset.width, dataset.height))
                    cm_img.save(os.path.join(hungarian_cd_save_path, "cd_{filename}.png".format(filename=post_name)))
                    np_cm_img = (np.asarray(cm_img)*1).astype('int')
                    cm_img.putpalette(nuclei_palette)

                    gt_cd = np.array(Image.open(files["masks"])).astype('int')

                    training_metrics(
                        target=torch.from_numpy(gt_cd),
                        preds=torch.from_numpy(np_cm_img)
                    )
                    cd_branch_acc.update(
                        metrics.np_dice_coeff(np_cm_img[np.newaxis, :, :], gt_cd[np.newaxis, :, :]), 1)

                    cm_img.save(os.path.join(cd_save_path, "cd_{filename}.png".format(filename=post_name)))

                    # Save CM from CD branch
                    # cv2.imwrite(
                    #     os.path.join(cd_save_path, "cd_{filename}.png".format(filename=filename)),
                    #     cm_img*255
                    # )
                    # cm_im = Image.fromarray((full_cm * 255).astype(np.uint8), mode='P')
                    # cm_im = cm_im.resize((dataset.width, dataset.height))
                    # cm_im.save(os.path.join(cd_save_path, "cd_{filename}.png".format(filename=filename)))

                    # # Calculate final CM
                    # cm_x_probs = np.multiply(full_x_probs, hg_map)
                    # cm_y_probs = np.multiply(full_y_probs, hg_map)
                    #
                    # hg_prob = np.maximum(cm_x_probs, cm_y_probs)
                    # hg_probs.append(hg_prob)
                    #
                    # # final_prob = hg_prob * cm_weights[0] + cm_probs[i, 0, ...] * cm_weights[1]
                    # final_prob = np.maximum(hg_prob, full_cm_probs)
                    # final_probs.append(final_prob)
                    #
                    # final_map = (final_prob >= threshold) * 255
                    # final_map = Image.fromarray(final_map.astype(np.uint8), mode='P')
                    # final_map = final_map.resize((dataset.width, dataset.height))
                    # final_map.save(os.path.join(final_cd_path, "final_{filename}.png".format(filename=filename)))

            #hg_probs = np.array(hg_probs)
            #final_probs = np.array(final_probs)

            #hungarian_branch_acc.update(metrics.np_dice_coeff((hg_probs >= threshold)*1, cd_labels.cpu().numpy()), hg_probs.shape[0])
            #final_acc.update(metrics.np_dice_coeff((final_probs >= threshold)*1, cd_labels.cpu().numpy()), final_probs.shape[0])

    values = training_metrics.compute()
    print(values)
    print("CD Branch dice: {:.4f}".format(cd_branch_acc.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate ResUnet")
    parser.add_argument("--mode", default=1)
    parser.add_argument("--pretrain", type=str)
    parser.add_argument("--savepath", type=str)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--batchsize", default=8, type=int)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="yaml file for configuration"
    )
    parser.add_argument("--device", default="cuda:0", type=str, help="Device ID")


    args = parser.parse_args()

    device = torch.device(args.device)

    weights = [1.0]
    if int(args.mode) == 1:
        weights = [1.0, 0.1]
    elif int(args.mode) == 2:
        weights = [1.0, 0.1, 0.05]

    hp = HParam(args.config)
    main(hp, int(args.mode), weights, device, args.split, args.pretrain, args.savepath, args.threshold, args.batchsize)