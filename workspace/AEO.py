import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelUnit(nn.Module):
    def __init__(self, in_channels, eps=1e-6):
        super(SobelUnit, self).__init__()
        self.eps = eps
        
        sobel_kernel_x = torch.tensor([[1, 0, -1],
                                       [2, 0, -2],
                                       [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = torch.tensor([[1, 2, 1],
                                       [0, 0, 0],
                                       [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)

        with torch.no_grad():
            self.sobel_x.weight.copy_(sobel_kernel_x.repeat(in_channels, 1, 1, 1))
            self.sobel_y.weight.copy_(sobel_kernel_y.repeat(in_channels, 1, 1, 1))

        for param in self.parameters():
            param.requires_grad = False

        self.group_norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)

    def forward(self, x):
        sobel_x = self.sobel_x(x)
        sobel_y = self.sobel_y(x)
        sobel_mag = torch.sqrt(sobel_x ** 2 + sobel_y ** 2 + self.eps)
        sobel_mag = self.group_norm(sobel_mag)

        return torch.tanh(sobel_mag)

class EdgeResidualUnit(nn.Module):
    def __init__(self, channels):
        super(EdgeResidualUnit, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='leaky_relu')
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        identity = x
        out = self.act(self.norm(self.conv1(x)))
        out = self.conv2(out)
        return out + identity

class EdgeEnhancementBlock(nn.Module):
    def __init__(self, in_channels):
        super(EdgeEnhancementBlock, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.init_norm = nn.GroupNorm(1, in_channels)

        self.sobel_unit = SobelUnit(in_channels)
        self.edge_residual_unit = EdgeResidualUnit(in_channels)

        self.laplacian = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                                   groups=in_channels, bias=False)

        laplacian_weight = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).repeat(in_channels, 1, 1, 1)

        self.laplacian.weight = nn.Parameter(laplacian_weight, requires_grad=False)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.initial_conv.weight)
        if self.initial_conv.bias is not None:
            nn.init.zeros_(self.initial_conv.bias)

    def forward(self, x):
        x = self.init_norm(self.initial_conv(x))
        sobel_edge = self.sobel_unit(x)

        weighted = x * (sobel_edge + 1e-6)
        out = self.edge_residual_unit(weighted)
        lap_out = self.laplacian(out)

        return lap_out
