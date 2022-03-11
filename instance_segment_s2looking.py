import os
import torch
import argparse
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from core.res_unet import ResUnet
from utils import metrics
from dataset.s2looking import *
from dataset.utils import *
from PIL import Image, ImagePalette
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.hparams import HParam
from utils.images import *
from utils.hungarian import *

def main(hp, mode, weights, trained_path, saved_path, threshold=0.5, batch_size=8, save_sub_mask=False):
    assert (0 <= mode < 3)
    assert os.path.isfile(trained_path)

    img1_save_path = os.path.join(saved_path, "img1")
    img2_save_path = os.path.join(saved_path, "img2")
    cd_save_path = os.path.join(saved_path, "cd")

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    if not os.path.exists(img1_save_path):
        os.makedirs(img1_save_path)

    if not os.path.exists(img2_save_path):
        os.makedirs(img2_save_path)

    if not os.path.exists(cd_save_path):
        os.makedirs(cd_save_path)

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
        additional_targets={'image0': 'image'}
    )

    dataset = S2Looking(hp.dset_dir, "val", transform)

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
            input1 = data['x'][:, 0, ...].cuda()
            input2 = data['x'][:, 1, ...].cuda()

            output1 = (model(input1).cpu().numpy() >= threshold) * 1
            output2 = (model(input2).cpu().numpy() >= threshold) * 1
            print(output1.shape)
            for i in range(output1.shape[0]):
                filename = dataset.files[idx * batch_size + i]
                filename = os.path.basename(filename['image1'])[:-4]

                masks1 = save_mask_and_contour(output1[i, 0, ...], output1[i, 1, ...], NUCLEI_PALETTE, os.path.join(img1_save_path, "is_{filename}.png".format(filename=filename)))
                masks2 = save_mask_and_contour(output2[i, 0, ...], output2[i, 1, ...], NUCLEI_PALETTE, os.path.join(img2_save_path, "is_{filename}.png".format(filename=filename)))
                print(masks1.shape)
                print(masks2.shape)
                masks1 = torch.from_numpy(masks1)
                masks2 = torch.from_numpy(masks2)
                # Hungarian algorithm
                cd_map = change_detection_map(masks1, masks2, 256, 256).numpy()
                im = Image.fromarray(cd_map, mode='P')
                im.save(os.path.join(cd_save_path, "{filename}.png".format(filename=filename)))

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

