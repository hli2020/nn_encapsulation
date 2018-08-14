from layers.cap_layer import CapConv, CapConv2, CapFC, CapLayer
from layers.OT_module import OptTrans
import torch.nn as nn


def _make_layer(inplanes, block, planes, block_num,
                stride=1, use_groupBN=False):
        """make resnet sub-layers"""
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = list()
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, int(block_num)):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def build_net(type, opts, num_classes):

    use_ot_loss = False  # TODO (low): should be put in the option.py

    module1, module2, module3, module4, final_cls = [], [], [], [], []
    module1_ot_loss, module2_ot_loss, module3_ot_loss, module4_ot_loss = [], [], [], []

    # DEFAULT SETTING FOR ENCAPNET
    if type == 'encapnet_set_OT':
        module1 = CapConv2(ch_in=32*1, ch_out=32*2, groups=32,
                           residual=[True, True], iter_N=2, no_downsample=True,
                           layerwise_skip_connect=False,
                           wider_main_conv=False,
                           manner='0')
        module2 = CapConv2(ch_in=32*2, ch_out=32*4, groups=32,
                           residual=[True, True], iter_N=2,
                           layerwise_skip_connect=False,
                           wider_main_conv=False,
                           manner='0')

        manner = '3' if opts.withCapRoute else '0'
        module3 = CapConv2(ch_in=32*4, ch_out=32*8, groups=32,
                           residual=[True, True], iter_N=2,
                           layerwise_skip_connect=False,
                           wider_main_conv=False,
                           manner=manner,
                           ot_choice='within')

        # several ablation studies here
        use_ot_loss = True
        group = 32 if opts.encapsulate_G else 1
        module3_ot_loss = OptTrans(ch_x=32*8, ch_y=32*8,
                                   spatial_x=8, spatial_y=8,
                                   remove_bias=opts.remove_bias,
                                   group=group, C_form=opts.C_form,
                                   no_bp_P_L=opts.no_bp_P_L,
                                   skip_critic=opts.skip_critic)

        module4 = CapConv2(ch_in=32*8, ch_out=32*16, groups=32,
                           residual=[True, True], iter_N=2,
                           layerwise_skip_connect=False,
                           wider_main_conv=False,
                           manner=manner,
                           ot_choice='within2')
        module4_ot_loss = OptTrans(ch_x=32*16, ch_y=32*8,
                                   spatial_x=4, spatial_y=8,
                                   remove_bias=opts.remove_bias,
                                   group=group, C_form=opts.C_form,
                                   no_bp_P_L=opts.no_bp_P_L,
                                   skip_critic=opts.skip_critic)

        # output after module4: bs, 32*16, 4, 4
        final_cls = CapFC(in_cap_num=32*4*4, out_cap_num=num_classes,
                          cap_dim=16, fc_manner='default')

    # DEFAULT SETTING FOR RESNET
    elif type == 'resnet_default':
        assert (opts.depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (opts.depth - 2) / 6
        block = Bottleneck if opts.depth >= 44 else BasicBlock
        stride_num, fc_num, pool_stride, pool_kernel = 1, 64, 1, 8

        inplanes = 16
        module1 = _make_layer(inplanes, block, 16, n)
        module2 = _make_layer(inplanes, block, 32, n, stride=2)
        module3 = _make_layer(inplanes, block, 64, n, stride=2)
        module4 = nn.AvgPool2d(pool_kernel, stride=pool_stride)
        final_cls = nn.Linear(fc_num, num_classes)

    # DEFAULT SETTING FOR RESNET
    elif type == 'capnet_default':
        module1 = nn.Sequential(*[
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        ])  # 32 spatial output
        module2 = nn.Sequential(*[
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        ])  # 16 spatial output
        module3 = nn.Sequential(*[
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        ])  # 8 spatial output

        module4 = CapLayer(opts, num_in_caps=opts.primary_cap_num * 8 * 8, in_dim=8,
                           num_out_caps=opts.primary_cap_num * 4 * 4, out_dim=16,
                           num_shared=opts.primary_cap_num, route=opts.route,
                           as_conv_output=True)
        final_cls = CapLayer(opts, num_in_caps=opts.primary_cap_num * 4 * 4, in_dim=16,
                             num_out_caps=num_classes, out_dim=16,
                             num_shared=opts.primary_cap_num, route=opts.route)

    return use_ot_loss, \
        module1, module2, module3, module4, final_cls, \
           module1_ot_loss, module2_ot_loss, module3_ot_loss, module4_ot_loss
