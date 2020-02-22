from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from math import ceil
from typing import List, Tuple

import torch
import numpy as np

from .utils.loading import load_model
from .config import CONFIG_RESNET_50
from .model.core import RetinaFaceModel
from .utils.inference import decode_bboxes, decode_landmarks, nms, round_value


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    @property
    def compact_form(self) -> Tuple[int, int]:
        return self.x, self.y


@dataclass(frozen=True)
class BoundingBox:
    left_top: Point
    right_bottom: Point


@dataclass(frozen=True)
class RetinaFaceNetPrediction:
    bbox: BoundingBox
    landmarks: List[Point]
    confidence: float


class RetinaFaceNet:

    @classmethod
    def initialize(cls,
                   weights_path: str,
                   use_gpu: bool = False,
                   confidence_threshold: float = 0.3,
                   top_k: int = 20,
                   nms_threshold: float = 0.4
                   ) -> RetinaFaceNet:
        model = RetinaFaceModel()
        model = load_model(
            model=model,
            weights_path=weights_path,
            use_gpu=use_gpu
        )
        return cls(
            model=model,
            device=torch.device("cuda" if use_gpu else "cpu"),
            confidence_threshold=confidence_threshold,
            top_k=top_k,
            nms_threshold=nms_threshold
        )

    __CHANNELS_STANDARDISATION_VALUES = (104, 117, 123)

    def __init__(self,
                 model: RetinaFaceModel,
                 device: torch.device,
                 confidence_threshold: float,
                 top_k: int,
                 nms_threshold: float
                 ):
        self.__model = model
        self.__device = device
        self.__confidence_threshold = confidence_threshold
        self.__top_k = top_k
        self.__nms_threshold = nms_threshold

    def infer(self, image: np.ndarray) -> List[RetinaFaceNetPrediction]:
        image_height, image_width = image.shape[:2]
        image = self.__standardise_input(image=image)
        bboxes, confidence, landmarks = self.__model(image)
        prior_boxes = self.__prepare_prior_boxes(
            image_height=image_height,
            image_width=image_width
        )
        bboxes = self.__decode_bboxes(
            bboxes=bboxes,
            input_tensor=image,
            prior_boxes=prior_boxes
        )
        confidence = confidence.squeeze(0).data.cpu().numpy()[:, 1]
        landmarks = self.__decode_landmarks(
            landmarks=landmarks,
            input_tensor=image,
            prior_boxes=prior_boxes,
        )
        bboxes, confidence, landmarks = self.__refine_results(
            bboxes=bboxes,
            confidence=confidence,
            landmarks=landmarks
        )
        return self.__wrap_results(
            bboxes=bboxes,
            confidence=confidence,
            landmarks=landmarks
        )

    def __prepare_prior_boxes(self,
                              image_height: int,
                              image_width: int,
                              ) -> torch.Tensor:
        steps = CONFIG_RESNET_50['steps']
        feature_maps = [
            (ceil(image_height / step), ceil(image_width / step))
            for step in steps
        ]
        anchors = []
        for k, f in enumerate(feature_maps):
            min_sizes = CONFIG_RESNET_50['min_sizes'][k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    step_x = min_size / image_width
                    step_y = min_size / image_height
                    dense_cx = [x * steps[k] / image_width for x in [j + 0.5]]
                    dense_cy = [y * steps[k] / image_height for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, step_x, step_y]
        output = torch.Tensor(anchors).view(-1, 4)
        if CONFIG_RESNET_50['clip']:
            output.clamp_(max=1, min=0)
        output = output.to(self.__device)
        return output

    def __standardise_input(self, image: np.ndarray) -> torch.Tensor:
        image = np.float32(image)
        image -= RetinaFaceNet.__CHANNELS_STANDARDISATION_VALUES
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0)
        return image.to(self.__device)

    def __decode_bboxes(self,
                        bboxes: torch.Tensor,
                        input_tensor: torch.Tensor,
                        prior_boxes: torch.Tensor
                        ) -> np.ndarray:
        bboxes = decode_bboxes(
            locations=bboxes.data.squeeze(0),
            priors=prior_boxes.data,
            variances=CONFIG_RESNET_50['variance']
        )
        shape_list = [
            input_tensor.shape[1], input_tensor.shape[0]
        ] * 2
        image_scale = torch.Tensor(shape_list)
        image_scale = image_scale.to(self.__device)
        bboxes = bboxes * image_scale
        return bboxes.cpu().numpy()

    def __decode_landmarks(self,
                           landmarks: torch.Tensor,
                           input_tensor: torch.Tensor,
                           prior_boxes: torch.Tensor,
                           ) -> np.ndarray:
        landmarks = decode_landmarks(
            predictions=landmarks.data.squeeze(0),
            priors=prior_boxes.data,
            variances=CONFIG_RESNET_50['variance']
        )
        scale_tensor_content = [
            input_tensor.shape[3], input_tensor.shape[2]
        ] * 5
        landmarks_scale = torch.Tensor(scale_tensor_content)
        landmarks_scale = landmarks_scale.to(self.__device)
        landmarks = landmarks * landmarks_scale
        return landmarks.cpu().numpy()

    def __refine_results(self,
                         bboxes: np.ndarray,
                         confidence: np.ndarray,
                         landmarks: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        to_take = np.where(confidence > self.__confidence_threshold)[0]
        bboxes = bboxes[to_take]
        confidence = confidence[to_take]
        landmarks = landmarks[to_take]
        order = confidence.argsort()[::-1][:self.__top_k]
        bboxes = bboxes[order]
        confidence = confidence[order]
        landmarks = landmarks[order]
        detections = np.hstack((bboxes, confidence[:, np.newaxis]))
        detections = detections.astype(np.float32,copy=False)
        to_keep = nms(
            detections=detections,
            threshold=self.__nms_threshold
        )
        return bboxes[to_keep], confidence[to_keep], landmarks[to_keep]

    def __wrap_results(self,
                       bboxes: np.ndarray,
                       confidence: np.ndarray,
                       landmarks: np.ndarray
                       ) -> List[RetinaFaceNetPrediction]:
        predictions = []
        iterable = zip(bboxes, confidence, landmarks)
        for bbox, confidence_score, example_landmarks in iterable:
            wrapped_bbox = self.__wrap_bbox(bbox=bbox)
            wrapped_landmarks = self.__wrap_landmarks(
                landmarks=example_landmarks
            )
            prediction = RetinaFaceNetPrediction(
                bbox=wrapped_bbox,
                landmarks=wrapped_landmarks,
                confidence=confidence_score
            )
            predictions.append(prediction)
        return predictions

    def __wrap_bbox(self, bbox: np.ndarray) -> BoundingBox:
        bbox = list(map(round_value, bbox.tolist()))
        left_top = Point(x=bbox[0], y=bbox[1])
        right_bottom = Point(x=bbox[2], y=bbox[3])
        return BoundingBox(left_top=left_top, right_bottom=right_bottom)

    def __wrap_landmarks(self, landmarks: np.ndarray) -> List[Point]:
        landmarks = list(map(round_value, landmarks.tolist()))
        result = []
        for i in range(5):
            landmark = Point(x=landmarks[2*i], y=landmarks[2*i+1])
            result.append(landmark)
        return result

