"""
Implementation Model ESRGAN

Author = FOZAME ENDEZOUMOU ARMAND BRYAN
"""

import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act, slope = 2, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            bias = True,
            **kwargs
        )

        self.act = nn.LeakyReLU(negative_slope = slope, inplace = True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))
    
class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor = 2, mode = "bicubic", slope = 0.2):
        super().__init__()
        self.upsample_block = nn.Upsample(scale_factor = scale_factor, 
                                          mode = mode)
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = in_channels,
                              kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.act = nn.LeakyReLU(negative_slope = slope, inplace = True)
    
    def forward(self, x):
        return self.act(self.conv(self.upsample_block(x)))

class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, channels = 32, residual_beta = 0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()

        for i in range(5):
            self.blocks.append(
                ConvBlock(
                    in_channels + channels * i,
                    channels if i<=3 else in_channels,
                    act = True if i<=3 else False,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
                )
            )
    
    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], dim = 1)
        
        return self.residual_beta * out + x 

class StackDRB(nn.Module):
    def __init__(self, in_channels, residual_beta = 0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.stackrdb = nn.Sequential(
            *[DenseResidualBlock(in_channels) for _ in range(3)]
        )

    def forward(self, x):
        return self.stackrdb(x) * self.residual_beta + x
    
class Generator(nn.Module):
    def __init__(self, in_channels = 3, num_channels = 64, num_blocks = 23, slope = 0.2):
        super().__init__()
        self.initial = nn.Conv2d(
            in_channels, num_channels, kernel_size = 3,
            stride = 1, padding = 1, bias = True
        )

        self.residuals = nn.Sequential(
            *[StackDRB(num_channels) for _ in range(num_blocks)]
        )

        self.conv = nn.Conv2d(
            num_channels, num_channels, kernel_size = 3,
            stride = 1, padding = 1, bias = True
        )

        self.upsamples = nn.Sequential(
            UpSampleBlock(num_channels),
            UpSampleBlock(num_channels),
        )

        self.final = nn.Sequential(
            nn.Conv2d(num_channels, num_channels,
                      kernel_size = 3, stride = 1,
                      padding = 1, bias = True),
            nn.LeakyReLU(negative_slope = slope, inplace = True),
            nn.Conv2d(num_channels, in_channels, kernel_size = 3,
                      stride = 1, padding = 1, bias = True)

        )

    def forward(self, x):
        initial = self.initial(x)
        x = self.conv(self.residuals(initial)) + initial
        x = self.upsamples(x)
        return self.final(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, slope = 0.2, features = [64,64,128,128,256,256,512,512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels = in_channels,
                    out_channels = feature,
                    kernel_size = 3,
                    stride = 1 + idx % 2,
                    padding = 1,
                    act = True
                ),
            )
            in_channels = feature
        
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6,6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(negative_slope = slope , inplace = True),
            nn.Linear(1024, 1)
        )
    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)

def initialize_weights(model, scale = 0.1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale
        
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale







        
