import os
import torch
import argparse
import numpy as np
import albumentations as albums
from albumentations.pytorch import ToTensorV2
from core.mixedmodel_cd_based import DependentResUnetMultiDecoder
from utils import metrics
from dataset.s2looking_allmask import S2LookingAllMask
from dataset.utils import *
from PIL import Image, ImagePalette
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.hparams import HParam
from utils.images import *
from utils.hungarian import *


def main(hp, mode, weights, trained_path, saved_path, threshold=0.5, batch_size=8, save_sub_mask=False,
         cm_weights=[0.5, 0.5]):
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

    model = DependentResUnetMultiDecoder().cuda()
    checkpoint = torch.load(trained_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if save_sub_mask:
        n_masks = mode + 1
    else:
        n_masks = 1

    transform = albums.Compose([
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

    dataset = S2LookingAllMask(hp.cd_dset_dir, "val", transform)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=2, shuffle=False
    )

    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()

    criterion = metrics.BCEDiceLoss(weight=[0.1, 0.9])

    loader = tqdm(dataloader, desc="Evaluating")

    NUCLEI_PALETTE = ImagePalette.random()

    with torch.no_grad():
        for (idx, data) in enumerate(loader):
            cd_i1 = data['x'].cuda()
            cd_i2 = data['y'].cuda()
            outputs = model(cd_i1, cd_i2)

            cm_prob = outputs['cm'].cpu().numpy()
            x_prob = outputs['x'].cpu().numpy()
            y_prob = outputs['y'].cpu().numpy()

            cm = (cm_prob >= threshold) * 1
            x = (x_prob >= threshold) * 1
            y = (y_prob >= threshold) * 1

            for i in range(cm.shape[0]):
                filename = dataset.files[idx * batch_size + i]
                filename = os.path.basename(filename['image1'])[:-4]

                masks1 = save_mask_and_contour(
                    x[i, 0, ...], x[i, 1, ...], NUCLEI_PALETTE,
                    os.path.join(img1_save_path, "is_{filename}.png".format(filename=filename)))\
                    .astype(int)
                masks2 = save_mask_and_contour(
                    y[i, 0, ...], y[i, 1, ...], NUCLEI_PALETTE,
                    os.path.join(img2_save_path, "is_{filename}.png".format(filename=filename))).\
                    astype(int)

                masks1 = torch.from_numpy(masks1)
                masks2 = torch.from_numpy(masks2)
                # Hungarian algorithm
                hungarian_cd_map = change_detection_map(masks1, masks2, 256, 256)
                hungarian_cd_image = (hungarian_cd_map * 255).astype(np.uint8)

                mask_color_1 = convert_to_color_map(masks1)
                mask_color_2 = convert_to_color_map(masks2)
                plot_and_save(mask_color_1, mask_color_2, hungarian_cd_image,
                              os.path.join(hungarian_cd_save_path, "{filename}.png".format(filename=filename)))

                cm_im = Image.fromarray((cm[i, 0, ...] * 255).astype(np.uint8), mode='P')
                cm_im.save(os.path.join(cd_save_path, "{filename}.png".format(filename=filename)))

                # Calculate final map
                cm_x_prob = np.multiply(x_prob[i, 0, ...], hungarian_cd_map)
                cm_y_prob = np.multiply(y_prob[i, 0, ...], hungarian_cd_map)

                final_cm_map = np.maximum(cm_x_prob, cm_y_prob)
                final_cm_map = final_cm_map * cm_weights[0] + cm_prob[i, 0, ...] * cm_weights[1]
                final_cm_map = (final_cm_map >= threshold) * 255
                final_cm_map = Image.fromarray(final_cm_map.astype(np.uint8), mode='P')
                final_cm_map.save(os.path.join(final_cd_path, "{filename}.png".format(filename=filename)))

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
