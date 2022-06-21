import argparse
import os

import cv2
import numpy as np
import torch
import torchmetrics
from PIL import Image
from tqdm import tqdm

from utils import metrics
from utils.hungarian import change_detection_map


def main(outputs1, outputs2, gts):
    hungarian_branch_acc = metrics.MetricTracker()
    training_metrics = torchmetrics.MetricCollection(
        {
            "Dice": torchmetrics.Dice(average='none', num_classes=2),
        },
        prefix='test_'
    )

    op = os.listdir(outputs1)
    gp = os.listdir(gts)

    op.sort()
    gp.sort()

    for output, gt in tqdm(zip(op, gp)):
        masks1 = Image.open(os.path.join(outputs1, output))
        masks2 = Image.open(os.path.join(outputs2, 'mask2' + output[5:]))

        masks1 = np.array(masks1.resize((1024, 1024))).astype(int)
        masks2 = np.array(masks2.resize((1024, 1024))).astype(int)
        # masks1 = cv2.cvtColor(masks1, cv2.COLOR_BGR2RGB)
        # masks2 = cv2.cvtColor(masks2, cv2.COLOR_BGR2RGB)

        # masks1 = cv2.resize(masks1, (1024, 1024))
        # masks2 = cv2.resize(masks2, (1024, 1024))

        masks1 = torch.from_numpy(masks1)
        masks2 = torch.from_numpy(masks2)

        #gt_cd = (np.array(Image.open(files["mask"])) / 255.0).astype('int')

        gt_cd = (np.array(Image.open(os.path.join(gts, gt))) / 255.0).astype('int')
        #gt_cd = cv2.imread(os.path.join(gts, gt))[:, :, 0]

        hg_map = change_detection_map(masks1, masks2, 1024, 1024)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        training_metrics(
            target=torch.from_numpy(gt_cd),
            preds=torch.from_numpy(hg_map)
        )
        hungarian_branch_acc.update(metrics.np_dice_coeff(hg_map[np.newaxis, :, :], gt_cd[np.newaxis, :, :]), 1)

    #values = training_metrics.compute()
    #print(values)
    print(hungarian_branch_acc.avg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate ResUnet")
    parser.add_argument("--o1", default="D:/HOC/Thesis result/S2Looking/alabama-segmentation/img1")
    parser.add_argument("--o2", default="D:/HOC/Thesis result/S2Looking/alabama-segmentation/img2")
    parser.add_argument("--g", default="D:/HOC/Thesis result/S2Looking/S2Looking/test/label")
    args = parser.parse_args()

    main(args.o1, args.o2, args.g)

