from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F

from retina_face_net.config import CONFIG_RESNET_50
from .building_blocks import FPN, SSH, ClassHead, BboxHead, LandmarkHead, \
    FPNResult

ExtractedFeatures = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
RetinaFaceResult = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class RetinaFaceModel(nn.Module):

    def __init__(self):
        super(RetinaFaceModel, self).__init__()
        backbone = models.resnet50(pretrained=False)
        self.body = _utils.IntermediateLayerGetter(
            model=backbone,
            return_layers=CONFIG_RESNET_50['backbone_layers_to_extract']
        )
        in_channels_stage2 = CONFIG_RESNET_50['in_channel']
        in_channels_list = (
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8
        )
        out_channels = CONFIG_RESNET_50['out_channel']
        self.fpn = FPN(
            in_channels_list=in_channels_list,
            out_channels=out_channels
        )
        self.ssh1 = SSH(in_channels=out_channels, out_channels=out_channels)
        self.ssh2 = SSH(in_channels=out_channels, out_channels=out_channels)
        self.ssh3 = SSH(in_channels=out_channels, out_channels=out_channels)
        self.ClassHead = self.__make_class_head(
            heads_num=3,
            in_channels=CONFIG_RESNET_50['out_channel']
        )
        self.BboxHead = self.__make_bbox_head(
            heads_num=3,
            in_channels=CONFIG_RESNET_50['out_channel']
        )
        self.LandmarkHead = self.__make_landmark_head(
            heads_num=3,
            in_channels=CONFIG_RESNET_50['out_channel']
        )

    def forward(self, inputs):
        out = self.body(inputs)
        fpn = self.fpn(out)
        features = self.__extract_features(fpn_output=fpn)
        return self.__prepare_output(features=features)

    def __make_class_head(self,
                          heads_num: int,
                          in_channels: int,
                          anchor_number: int = 2
                          ) -> nn.Module:
        class_head = nn.ModuleList()
        for i in range(heads_num):
            class_head.append(ClassHead(in_channels, anchor_number))
        return class_head

    def __make_bbox_head(self,
                         heads_num: int,
                         in_channels: int,
                         anchor_num: int = 2
                         ) -> nn.Module:
        bbox_head = nn.ModuleList()
        for i in range(heads_num):
            bbox_head.append(BboxHead(in_channels, anchor_num))
        return bbox_head

    def __make_landmark_head(self,
                             heads_num: int,
                             in_channels: int,
                             anchor_num: int = 2
                             ) -> nn.Module:
        landmark_head = nn.ModuleList()
        for i in range(heads_num):
            landmark_head.append(LandmarkHead(in_channels, anchor_num))
        return landmark_head

    def __extract_features(self,
                           fpn_output: FPNResult
                           ) -> ExtractedFeatures:
        feature1 = self.ssh1(fpn_output[0])
        feature2 = self.ssh2(fpn_output[1])
        feature3 = self.ssh3(fpn_output[2])
        return feature1, feature2, feature3

    def __prepare_output(self,
                         features: ExtractedFeatures
                         ) -> RetinaFaceResult:

        bbox_regression = self.__prepare_bbox_regression(features=features)
        confidence = self.__prepare_confidence_results(
            features=features
        )
        landmark_regression = self.__prepare_landmark_regression(
            features=features
        )
        return bbox_regression, confidence, landmark_regression

    def __prepare_bbox_regression(self,
                                  features: ExtractedFeatures
                                  ) -> torch.Tensor:
        bbox_regression = [
            self.BboxHead[i](feature) for i, feature in enumerate(features)
        ]
        return torch.cat(bbox_regression, dim=1)

    def __prepare_confidence_results(self,
                                     features: ExtractedFeatures
                                     ) -> torch.Tensor:
        confidence = [
            self.ClassHead[i](feature) for i, feature in enumerate(features)
        ]
        confidence = torch.cat(confidence, dim=1)
        return F.softmax(confidence, dim=-1)

    def __prepare_landmark_regression(self,
                                      features: ExtractedFeatures
                                      ) -> torch.Tensor:
        landmark_regression = [
            self.LandmarkHead[i](feature)
            for i, feature in enumerate(features)
        ]
        return torch.cat(landmark_regression, dim=1)
