# from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# from retina_face_net.errors import ModelBuildingConstraintError
# from retina_face_net.model.layers import conv_1x1_bn_leaky_relu, \
#     conv_3x3_bn_leaky_relu, conv_bn
from .layers import conv_bn1X1, conv_bn, conv_bn_no_relu

FPNResult = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


# class FPN(nn.Module):
#
#     def __init__(self,
#                  in_channels_list: Tuple[int, int, int],
#                  out_channels: int):
#         super(FPN, self).__init__()
#         leaky_relu_negative_slope = 0.0 if out_channels <= 64 else 0.1
#         feature_extraction_conv = partial(
#             conv_1x1_bn_leaky_relu,
#             out_channels=out_channels,
#             leaky_relu_negative_slope=leaky_relu_negative_slope
#         )
#         self.output1 = feature_extraction_conv(
#             in_channels=in_channels_list[0]
#         )
#         self.output2 = feature_extraction_conv(
#             in_channels=in_channels_list[1]
#         )
#         self.output3 = feature_extraction_conv(
#             in_channels=in_channels_list[2]
#         )
#         merge_conv = partial(
#             conv_3x3_bn_leaky_relu,
#             in_channels=out_channels,
#             out_channels=out_channels,
#             leaky_relu_negative_slope=leaky_relu_negative_slope
#         )
#         self.merge1 = merge_conv()
#         self.merge2 = merge_conv()
#
#     def forward(self, x: torch.Tensor) -> FPNResult:
#         x = list(x.values())
#         output1 = self.output1(x[0])
#         output2 = self.output2(x[1])
#         output3 = self.output3(x[2])
#         output2 = self.__fuse(
#             larger_output=output2,
#             smaller_output=output3
#         )
#         output2 = self.merge2(output2)
#         output1 = self.__fuse(
#             larger_output=output1,
#             smaller_output=output2
#         )
#         output1 = self.merge1(output1)
#         return output1, output2, output3
#
#     def __fuse(self,
#                larger_output: torch.Tensor,
#                smaller_output: torch.Tensor
#                ):
#         smaller_output_upscalled = F.interpolate(
#             smaller_output,
#             size=[larger_output.size(2), larger_output.size(3)],
#             mode="nearest"
#         )
#         return larger_output + smaller_output_upscalled
#
#
# class SSH(nn.Module):
#
#     __EXCEPTION_MSG = \
#         "SSH building block requires output channels to be divisible by 4."
#
#     def __init__(self, in_channels: int, out_channels: int):
#         super(SSH, self).__init__()
#         if out_channels % 4 != 0:
#             raise ModelBuildingConstraintError(SSH.__EXCEPTION_MSG)
#         leaky_relu_negative_slope = 0.0 if out_channels <= 64 else 0.1
#         self.conv3X3 = conv_bn(
#             in_channels=in_channels,
#             out_channels=out_channels // 2,
#             kernel_size=3
#         )
#         self.conv5X5_1 = conv_3x3_bn_leaky_relu(
#             in_channels=in_channels,
#             out_channels=out_channels // 4,
#             leaky_relu_negative_slope=leaky_relu_negative_slope
#         )
#         self.conv5X5_2 = conv_bn(
#             in_channels=out_channels // 4,
#             out_channels=out_channels // 4,
#             kernel_size=3
#         )
#         self.conv7X7_2 = conv_3x3_bn_leaky_relu(
#             in_channels=out_channels // 4,
#             out_channels=out_channels // 4,
#             leaky_relu_negative_slope=leaky_relu_negative_slope
#         )
#         self.conv7x7_3 = conv_bn(
#             in_channels=out_channels // 4,
#             out_channels=out_channels // 4,
#             kernel_size=3
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         conv3X3 = self.conv3X3(x)
#         conv5X5_1 = self.conv5X5_1(x)
#         conv5X5 = self.conv5X5_2(conv5X5_1)
#         conv7X7_2 = self.conv7X7_2(conv5X5_1)
#         conv7X7 = self.conv7x7_3(conv7X7_2)
#         out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
#         out = F.relu(out)
#         return out


class SSH(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSH, self).__init__()
        assert out_channels % 4 == 0
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channels, out_channels // 2, stride=1)

        self.conv5X5_1 = conv_bn(in_channels, out_channels // 4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channels // 4, out_channels // 4, stride=1)

        self.conv7X7_2 = conv_bn(out_channels // 4, out_channels // 4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channels // 4, out_channels // 4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out


class ClassHead(nn.Module):

    def __init__(self, in_channels: int, num_anchors: int):
        super(ClassHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_anchors * 2,
            kernel_size=(1, 1),
            stride=1,
            padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):

    def __init__(self, in_channels: int, num_anchors: int):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_anchors * 4,
            kernel_size=(1, 1),
            stride=1,
            padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):

    def __init__(self, in_channels: int, num_anchors: int):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_anchors * 10,
            kernel_size=(1, 1),
            stride=1,
            padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)
