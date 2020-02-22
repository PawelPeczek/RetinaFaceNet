from typing import List

import numpy as np
import torch


def decode_bboxes(locations: torch.Tensor,
                  priors: torch.Tensor,
                  variances: List[float]
                  ) -> torch.Tensor:
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        locations (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: Variances of prior boxes
    Return:
        decoded bounding box predictions
    """
    boxes = (
        priors[:, :2] + locations[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(locations[:, 2:] * variances[1])
    )
    boxes = torch.cat(boxes, 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landmarks(predictions: torch.Tensor,
                     priors: torch.Tensor,
                     variances: List[float]
                     ) -> torch.Tensor:
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        predictions (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landmarks = (
        priors[:, :2] + predictions[:, :2] * variances[0] * priors[:, 2:],
        priors[:, :2] + predictions[:, 2:4] * variances[0] * priors[:, 2:],
        priors[:, :2] + predictions[:, 4:6] * variances[0] * priors[:, 2:],
        priors[:, :2] + predictions[:, 6:8] * variances[0] * priors[:, 2:],
        priors[:, :2] + predictions[:, 8:10] * variances[0] * priors[:, 2:],
    )
    return torch.cat(landmarks, dim=1)


# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
def nms(detections: np.ndarray, threshold: float) -> List[int]:
    """Pure Python NMS baseline."""
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]
    scores = detections[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        indices = np.where(ovr <= threshold)[0]
        order = order[indices + 1]
    return keep


def round_value(value: float) -> int:
    return int(round(value))
