import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_


class EpiEncoder(nn.Module):
    """ Learns an encoding of epipolar images when presented as a 2D grid of epipolar slices. """
 
    def __init__(self, direction, tilesize):
        """
        Arguments:
            - direction: "vertical" or "horizontal"                                                # TODO: this
            - tilesize: width (for vertical) or height (for horizontal) of epipolar slice
        Returns:
            - neural layer that encodes an EPI
        """
        
        assert direction in ['vertical']
        
        super(EpiEncoder, self).__init__()

        self.direction = direction
        

        if direction == 'vertical':
            pad_top = tilesize//2
            self.conv1 = nn.Conv2d(1, 16, kernel_size=tilesize, stride=(1, tilesize), padding=(pad_top, 0))

        self.relu1 = nn.ReLU(inplace=True)

    
    def forward(self, lf_formatted, lf_stacked):
        """
        Arguments:
            - lf_formatted: the tiled epipolar image    [B, 1, H, W*tilesize] or [B, 1, H*tilesize, W]
            - lf_stacked: the grid of images stacked on the colour-channel   [B, N, H, W]
        Returns:
            - encoding: the encoded light field concatenated with the stacked image-grid   [B, N+32, H, W]
        """
        inp_height, inp_width = lf_formatted.shape[2:]

        x = self.conv1(lf_formatted)
        x = self.relu1(x)

        if self.direction == 'vertical':
            x = x[:, :, 0:inp_height, :]
        elif self.direction == 'horizontal':
            x = x[:, :, :, 0:inp_width]

        x = torch.cat([x, lf_stacked], dim=1)

        return x 
        

""" Demo showing operation of encoder """
if __name__ == "__main__":
    net = EpiEncoder('vertical', 8)
    d = torch.rand([4, 1, 200, 200*8])
    g = torch.rand([4, 8, 200, 200])
    y = net(d, g)
    print(y.shape)