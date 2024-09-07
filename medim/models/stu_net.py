import torch
from torch import nn
from ._registry import register_model
from ._pretrain import load_pretrained_weights


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.deep_supervision = True


class BasicResBlock(nn.Module):

    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1,
                 use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels,
                               output_channels,
                               kernel_size,
                               stride=stride,
                               padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class Upsample_Layer_nearest(nn.Module):

    def __init__(self, input_channels, output_channels, pool_op_kernel_size, mode='nearest'):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


DEFAULT_STRIDES = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 2]]


class STUNet(nn.Module):

    def __init__(self,
                 input_channels=1,
                 num_classes=105,
                 depth=[1, 1, 1, 1, 1, 1],
                 dims=[32, 64, 128, 256, 512, 512],
                 pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 enable_deep_supervision=False):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.final_nonlin = lambda x: x
        self.decoder = Decoder()
        self.decoder.deep_supervision = enable_deep_supervision
        self.upscale_logits = False

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        num_pool = len(pool_op_kernel_sizes)

        assert num_pool == len(dims) - 1

        # encoder
        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(
            BasicResBlock(input_channels,
                          dims[0],
                          self.conv_kernel_sizes[0],
                          self.conv_pad_sizes[0],
                          use_1x1conv=True), *[
                              BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0],
                                            self.conv_pad_sizes[0]) for _ in range(depth[0] - 1)
                          ])
        self.conv_blocks_context.append(stage)
        for d in range(1, num_pool + 1):
            stage = nn.Sequential(
                BasicResBlock(dims[d - 1],
                              dims[d],
                              self.conv_kernel_sizes[d],
                              self.conv_pad_sizes[d],
                              stride=self.pool_op_kernel_sizes[d - 1],
                              use_1x1conv=True), *[
                                  BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d],
                                                self.conv_pad_sizes[d])
                                  for _ in range(depth[d] - 1)
                              ])
            self.conv_blocks_context.append(stage)

        # upsample_layers
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            upsample_layer = Upsample_Layer_nearest(dims[-1 - u], dims[-2 - u],
                                                    pool_op_kernel_sizes[-1 - u])
            self.upsample_layers.append(upsample_layer)

        # decoder
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            stage = nn.Sequential(
                BasicResBlock(dims[-2 - u] * 2,
                              dims[-2 - u],
                              self.conv_kernel_sizes[-2 - u],
                              self.conv_pad_sizes[-2 - u],
                              use_1x1conv=True),
                *[
                    BasicResBlock(dims[-2 - u], dims[-2 - u], self.conv_kernel_sizes[-2 - u],
                                  self.conv_pad_sizes[-2 - u]) for _ in range(depth[-2 - u] - 1)
                ])
            self.conv_blocks_localization.append(stage)

        # outputs
        self.seg_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(nn.Conv3d(dims[-2 - ds], num_classes, kernel_size=1))

        self.upscale_logits_ops = []
        for usl in range(num_pool - 1):
            self.upscale_logits_ops.append(lambda x: x)

    def forward(self, x):
        skips = []
        seg_outputs = []

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self.decoder.deep_supervision:
            return tuple([seg_outputs[-1]] + [
                i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])
            ])
        else:
            return seg_outputs[-1]


@register_model("STU-Net-S")
def create_stunet_small(
    pretrained: bool = False,
    checkpoint_path: str = '',
    strides=DEFAULT_STRIDES,
    **kwargs,
) -> STUNet:
    model = STUNet(depth=[1] * 6,
                   dims=[16 * x for x in [1, 2, 4, 8, 16, 16]],
                   pool_op_kernel_sizes=strides,
                   conv_kernel_sizes=[[3, 3, 3]] * 6,
                   enable_deep_supervision=False,
                   **kwargs)
    if (pretrained):
        load_pretrained_weights(model, checkpoint_path)

    return model


@register_model("STU-Net-B")
def create_stunet_base(
    pretrained: bool = False,
    checkpoint_path: str = '',
    strides=DEFAULT_STRIDES,
    **kwargs,
) -> STUNet:
    model = STUNet(depth=[1] * 6,
                   dims=[32 * x for x in [1, 2, 4, 8, 16, 16]],
                   pool_op_kernel_sizes=strides,
                   conv_kernel_sizes=[[3, 3, 3]] * 6,
                   enable_deep_supervision=False,
                   **kwargs)
    if (pretrained):
        load_pretrained_weights(model, checkpoint_path)

    return model


@register_model("STU-Net-L")
def create_stunet_large(
    pretrained: bool = False,
    checkpoint_path: str = '',
    strides=DEFAULT_STRIDES,
    **kwargs,
) -> STUNet:
    model = STUNet(depth=[2] * 6,
                   dims=[64 * x for x in [1, 2, 4, 8, 16, 16]],
                   pool_op_kernel_sizes=strides,
                   conv_kernel_sizes=[[3, 3, 3]] * 6,
                   enable_deep_supervision=False,
                   **kwargs)
    if (pretrained):
        load_pretrained_weights(model, checkpoint_path)

    return model


@register_model("STU-Net-H")
def create_stunet_huge(
    pretrained: bool = False,
    checkpoint_path: str = '',
    strides=DEFAULT_STRIDES,
    **kwargs,
) -> STUNet:
    model = STUNet(depth=[3] * 6,
                   dims=[96 * x for x in [1, 2, 4, 8, 16, 16]],
                   pool_op_kernel_sizes=strides,
                   conv_kernel_sizes=[[3, 3, 3]] * 6,
                   enable_deep_supervision=False,
                   **kwargs)
    if (pretrained):
        load_pretrained_weights(model, checkpoint_path)

    return model
