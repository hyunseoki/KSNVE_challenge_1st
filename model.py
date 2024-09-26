import torch
import torch.nn as nn


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, instance_norm=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=nn.InstanceNorm1d,
            ),
        )
        if instance_norm:
            self.block.append(nn.InstanceNorm1d(out_channels))
        self.block.append(nn.GELU())

    def forward(self, x):
        return self.block(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=nn.InstanceNorm1d,
            ),
            nn.InstanceNorm1d(out_channels),
            nn.GELU()
        )
        if dropout:
            self.block.append(nn.Dropout(0.5))

    def forward(self, x):
        x = self.block(x)

        return  x


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_filters=20):
        super().__init__()
        self.downsamples = nn.Sequential(
            Downsample(in_channels=in_channels, out_channels=n_filters),
            Downsample(in_channels=n_filters, out_channels=n_filters * 2),
            Downsample(in_channels=n_filters * 2, out_channels=n_filters * 4),
            Downsample(in_channels=n_filters * 4, out_channels=n_filters * 8),
            Downsample(in_channels=n_filters * 8, out_channels=n_filters * 16),
            Downsample(in_channels=n_filters * 16, out_channels=n_filters * 32),
        )

        self.upsamples = nn.Sequential(
            Upsample(in_channels=n_filters * 32, out_channels=n_filters * 16),
            Upsample(in_channels=n_filters * 32, out_channels=n_filters * 8),
            Upsample(in_channels=n_filters * 16, out_channels=n_filters * 4),
            Upsample(in_channels=n_filters * 8, out_channels=n_filters * 2),
            Upsample(in_channels=n_filters * 4, out_channels=n_filters),
        )

        self.last = nn.Sequential(
            nn.ConvTranspose1d(in_channels=n_filters * 2, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        init_weights(self)
        self.register_buffer('center_p', torch.zeros(n_filters * 6400))


    def freeze_decoder(self):
        self.upsamples.requires_grad_(False)
        self.last.requires_grad_(False)


    def encode(self, x):
        z = self.downsamples(x)
        return z.view(z.size(0), -1)


    def forward(self, x):
        left1 = self.downsamples[0](x) ## 16, 20, 6400
        left2 = self.downsamples[1](left1)
        left3 = self.downsamples[2](left2)
        left4 = self.downsamples[3](left3)
        left5 = self.downsamples[4](left4)
        center = self.downsamples[5](left5)

        x = torch.cat([self.upsamples[0](center), left5], 1)
        x = torch.cat([self.upsamples[1](x), left4], 1)
        x = torch.cat([self.upsamples[2](x), left3], 1)
        x = torch.cat([self.upsamples[3](x), left2], 1)
        x = torch.cat([self.upsamples[4](x), left1], 1)

        x = self.last(x)

        return x
