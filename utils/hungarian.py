import torch
import numpy as np
from torchvision.ops.boxes import box_area
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F


def balance_2_boxes(boxes1, boxes2):
    """
    If 2 masks have a diffenrence in numbers of objects, this function will add
    some "empty" bboxs to the less objects masks
    """
    one = torch.tensor([[0, 0, 0, 0]])
    if boxes1.shape[0] > boxes2.shape[0]:
        for i in range(boxes1.shape[0] - boxes2.shape[0]):
            boxes2 = torch.cat([boxes2, one], dim=0)
    elif boxes1.shape[0] < boxes2.shape[0]:
        for i in range(boxes2.shape[0] - boxes1.shape[0]):
            boxes1 = torch.cat([boxes1, one], dim=0)
    return boxes1, boxes2


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks
    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.
    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def turn_2_masks_to_boxes(masks1, masks2):
    boxes1 = masks_to_boxes(masks1)
    boxes2 = masks_to_boxes(masks2)
    boxes1, boxes2 = balance_2_boxes(boxes1, boxes2)
    return boxes1, boxes2


def box_xyxy_to_cxcywh(x, p_w, p_h):
    """Convert bbox from format [x0,y0,x1,y1] to [center_x, center_y, width, height] in the percent of the size of picture
    (it will be divided by the width and height of the picture p_w, p_h)
    """
    x0, y0, x1, y1 = x.unbind(-1)
    x0, y0, x1, y1 = x0 / p_w, y0 / p_h, x1 / p_w, y1 / p_h
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def boxes_xyxy_to_cxcywh(boxes, p_w, p_h):
    """Convert all bboxes of a picture from format [x0,y0,x1,y1] to [center_x, center_y, width, height] in the percent of the size of picture
      (it will be divided by the width and height of the picture p_w, p_h)
    """
    result = torch.zeros(1, 4)
    for i in range(boxes.shape[0]):
        box = torch.reshape(box_xyxy_to_cxcywh(boxes[i], p_w, p_h), (1, 4))
        result = torch.cat([result, box], dim=0)
    return result[1:, :]


def box_iou(boxes1, boxes2, error=1e-5):
    """Calculate the IoU of bboxes of 2 pictures
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union + error
    return iou, union


def xor_between_2_masks(masks1, masks2):
    result = torch.ones(max(masks1.shape[0], masks2.shape[0]), max(masks1.shape[0], masks2.shape[0])) * torch.numel(
        masks1[0])
    for i in range(masks1.shape[0]):
        for j in range(masks2.shape[0]):
            xor_mask = torch.bitwise_xor(masks1[i], masks2[j])
            result[i][j] = torch.sum(xor_mask)
    return result / torch.numel(masks1[0])


def l1_loss_between_2_boxes(boxes1, boxes2, p_w, p_h):  # (N, 4)
    boxes_c1, boxes_c2 = boxes_xyxy_to_cxcywh(boxes1, p_w, p_h), boxes_xyxy_to_cxcywh(boxes2, p_w, p_h)
    result = torch.zeros(boxes1.shape[0], boxes2.shape[0])
    for i in range(boxes1.shape[0]):
        for j in range(boxes2.shape[0]):
            result[i][j] = F.l1_loss(boxes_c1[i], boxes_c2[j], reduction='sum').item()
    return result


def loss_between_2_masks(masks1, masks2, p_w, p_h, alpha=1, beta=1, gamma=1):
    boxes1, boxes2 = turn_2_masks_to_boxes(masks1, masks2)
    iou, union = box_iou(boxes1, boxes2)
    l1 = l1_loss_between_2_boxes(boxes1, boxes2, p_w, p_h)
    xor_mask = xor_between_2_masks(masks1, masks2)
    return -1 * alpha * torch.log(iou) + beta * l1 + gamma * xor_mask


def hungarians_calc(loss_matrix, threshold=1.3):
    row_ind, col_ind = linear_sum_assignment(loss_matrix)
    for i in range(len(col_ind)):
        if loss_matrix[i][col_ind[i]] > threshold:
            col_ind[i] = -1
    return row_ind, col_ind


def change_detection_map(masks1, masks2, w, h):
    loss12 = loss_between_2_masks(masks1, masks2, w, h)
    row_ind, col_ind = hungarians_calc(loss12)
    error_bboxes1 = np.where(col_ind == -1)

    loss21 = loss_between_2_masks(masks2, masks1, w, h)
    row_ind, col_ind = hungarians_calc(loss21)
    error_bboxes2 = np.where(col_ind == -1)

    cd_map = np.zeros_like(masks1[0])
    for layer in error_bboxes1[0]:
        if layer < masks1.shape[0]:
            cd_map = cd_map | masks1[layer].numpy()
    for layer in error_bboxes2[0]:
        if layer < masks2.shape[0]:
            cd_map = cd_map | masks2[layer].numpy()

    return cd_map
