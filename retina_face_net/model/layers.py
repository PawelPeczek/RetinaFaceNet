import torch.nn as nn


# def conv_1x1_bn_leaky_relu(in_channels: int,
#                            out_channels: int,
#                            stride: int = 1,
#                            bias: bool = False,
#                            leaky_relu_negative_slope: float = 0.0
#                            ) -> nn.Module:
#     return conv_bn_leaky_relu(
#         in_channels=in_channels,
#         out_channels=out_channels,
#         kernel_size=1,
#         stride=stride,
#         padding=0,
#         bias=bias,
#         leaky_relu_negative_slope=leaky_relu_negative_slope
#     )
#
#
# def conv_3x3_bn_leaky_relu(in_channels: int,
#                            out_channels: int,
#                            stride: int = 1,
#                            bias: bool = False,
#                            leaky_relu_negative_slope: float = 0.0
#                            ) -> nn.Module:
#     return conv_bn_leaky_relu(
#         in_channels=in_channels,
#         out_channels=out_channels,
#         kernel_size=3,
#         stride=stride,
#         padding=1,
#         bias=bias,
#         leaky_relu_negative_slope=leaky_relu_negative_slope
#     )
#
#
# def conv_bn_leaky_relu(in_channels: int,
#                        out_channels: int,
#                        kernel_size: int,
#                        stride: int = 1,
#                        padding: int = 1,
#                        bias: bool = False,
#                        leaky_relu_negative_slope: float = 0.0
#                        ) -> nn.Module:
#     convolution = conv_bn(
#         in_channels=in_channels,
#         out_channels=out_channels,
#         kernel_size=kernel_size,
#         stride=stride,
#         padding=padding,
#         bias=bias
#     )
#     return nn.Sequential(
#         convolution,
#         nn.LeakyReLU(negative_slope=leaky_relu_negative_slope, inplace=True)
#     )
#
#
# def conv_bn(in_channels: int,
#             out_channels: int,
#             kernel_size: int,
#             stride: int = 1,
#             padding: int = 1,
#             bias: bool = False
#             ) -> nn.Module:
#     convolution = nn.Conv2d(
#         in_channels=in_channels,
#         out_channels=out_channels,
#         kernel_size=kernel_size,
#         stride=stride,
#         padding=padding,
#         bias=bias
#     )
#     return nn.Sequential(
#         convolution,
#         nn.BatchNorm2d(num_features=out_channels),
#     )


def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )
