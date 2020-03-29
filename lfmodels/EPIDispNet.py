import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_


def downsample_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )


def predict_disp(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.Sigmoid()
    )


def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )


def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]


class EPIDispNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, alpha=10, beta=0.01):
        super(EPIDispNet, self).__init__()

        self.alpha = alpha
        self.beta = beta

        conv_planes = [128, 256, 512, 512, 1024, 1024]
        self.conv1 = conv(in_channels, conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2], kernel_size=3)
        self.conv4 = downsample_conv(conv_planes[2], conv_planes[3], kernel_size=3)
        self.conv5 = downsample_conv(conv_planes[3], conv_planes[4], kernel_size=3)
        self.conv6 = conv(conv_planes[4], conv_planes[5], kernel_size=3)

        upconv_planes = [1024, 512, 256, 128]

        self.upconv1 = upconv(conv_planes[5], upconv_planes[0])
        self.iconv1 = conv(upconv_planes[0] + conv_planes[3], upconv_planes[0])

        self.upconv2 = upconv(upconv_planes[0], upconv_planes[1])
        self.iconv2 = conv(upconv_planes[1] + conv_planes[2], upconv_planes[1])

        self.upconv3 = conv(upconv_planes[1], upconv_planes[2])
        self.iconv3 = conv(upconv_planes[2] + conv_planes[1], upconv_planes[2])
        # self.predict_disp_halfsize = nn.Sequential(
        #     conv(upconv_planes[2], in_channels//2),
        #     upconv(in_channels//2, in_channels//2)
        # )

        self.upconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.iconv4 = conv(upconv_planes[3] + conv_planes[0], upconv_planes[3])
        self.output_layer = predict_disp(upconv_planes[3], in_channels)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def predict_disp(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)

        out_upconv1 = self.upconv1(out_conv6)
        concat1 = torch.cat((out_upconv1, out_conv4), 1)
        out_iconv1 = self.iconv1(concat1)

        out_upconv2 = self.upconv2(out_iconv1)
        concat2 = torch.cat((out_upconv2, out_conv3), 1)
        out_iconv2 = self.iconv2(concat2)

        out_upconv3 = self.upconv3(out_iconv2)
        concat3 = torch.cat((out_upconv3, out_conv2), 1)
        out_iconv3 = self.iconv3(concat3)

        out_upconv4 = crop_like(self.upconv4(out_iconv3), out_conv1)
        concat4 = torch.cat((out_upconv4, out_conv1), 1)
        out_iconv4 = self.iconv4(concat4)
        disp_full = self.output_layer(out_iconv4)
        return disp_full

    def forward(self, x):
        """
        Expects first 8 rows (batch[:, :, :8, :]) to be horizontal epi, next
        8 rows should be vertical epi
        """
        disp_horizontal = self.predict_disp(x[:, :, :8, :]).permute([0, 2, 1, 3])
        disp_vertical = self.predict_disp(x[:, :, 8:, :]).permute([0, 2, 3, 1])
        disp = torch.mean(torch.stack((disp_horizontal, disp_vertical)), 0)
        if self.training:
            return [disp]
        else:
            return disp

if __name__ == "__main__":
    net = EPIDispNet(128)
    horizontal = torch.rand(4, 128, 16, 128)
    y = net(horizontal)
