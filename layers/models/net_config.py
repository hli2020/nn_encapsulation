from layers.cap_layer import CapConv2, CapFC
from layers.OT_module import OptTrans
from layers.misc import connect_list
import torch.nn as nn
from layers.models.cifar.resnet import BasicBlock, Bottleneck


def _make_layer(self, block, planes, blocks, stride=1, use_groupBN=False):
        """make resnet sub-layers"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, int(blocks)):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


def build_net(type, opts):

    input_ch = 1 if opts.dataset == 'fmnist' or opts.dataset == 'mnist' else 3
    module1, module2, module3, module4, final_cls = [], [], [], [], []
    module1_ot_loss, module2_ot_loss, module3_ot_loss, module4_ot_loss = [], [], [], []
    connect_detail = connect_list[opts.connect_detail]

    if type == 'default':
        module1 = CapConv2(ch_in=32*1, ch_out=32*2, groups=32,
                           residual=connect_detail[0:2], iter_N=opts.cap_N,
                           no_downsample=True,
                           use_capBN=opts.use_capBN, skip_relu=opts.skip_relu,
                           layerwise_skip_connect=opts.layerwise,
                           wider_main_conv=opts.wider,
                           manner=opts.manner)
        module2 = CapConv2(ch_in=32*2, ch_out=32*4, groups=32,
                           residual=connect_detail[2:4], iter_N=opts.cap_N,
                           use_capBN=opts.use_capBN, skip_relu=opts.skip_relu,
                           layerwise_skip_connect=opts.layerwise,
                           wider_main_conv=opts.wider,
                           manner=opts.manner)
        module3 = CapConv2(ch_in=32*4, ch_out=32*8, groups=32,
                           residual=connect_detail[4:6], iter_N=opts.cap_N,
                           use_capBN=opts.use_capBN, skip_relu=opts.skip_relu,
                           layerwise_skip_connect=opts.layerwise,
                           wider_main_conv=opts.wider,
                           manner=opts.manner)
        module4 = CapConv2(ch_in=32*8, ch_out=32*16, groups=32,
                           residual=connect_detail[6:], iter_N=opts.cap_N,
                           use_capBN=opts.use_capBN, skip_relu=opts.skip_relu,
                           layerwise_skip_connect=opts.layerwise,
                           wider_main_conv=opts.wider,
                           manner=opts.manner)
        # output after module4: bs, 32*16, 4, 4
        final_cls = CapFC(in_cap_num=32*4*4, out_cap_num=10,
                          cap_dim=16, fc_manner=opts.fc_manner)
    elif type == 'default_super':
        module1 = CapConv2(ch_in=32*1, ch_out=32*2, groups=32,
                           residual=[True, True], iter_N=4, no_downsample=True,
                           more_skip=True,
                           layerwise_skip_connect=False,
                           wider_main_conv=False,
                           manner='3')
        module2 = CapConv2(ch_in=32*2, ch_out=32*4, groups=32,
                           residual=[True, True], iter_N=8,
                           more_skip=True,
                           layerwise_skip_connect=True,
                           wider_main_conv=False,
                           manner='3')
        module3 = CapConv2(ch_in=32*4, ch_out=32*8, groups=32,
                           residual=[True, True], iter_N=4,
                           more_skip=True,
                           layerwise_skip_connect=False,
                           wider_main_conv=False,
                           manner='3')
        module4 = CapConv2(ch_in=32*8, ch_out=32*16, groups=32,
                           residual=[True, True], iter_N=8,
                           more_skip=True,
                           layerwise_skip_connect=True,
                           wider_main_conv=False,
                           manner='3')
        # output after module4: bs, 32*16, 4, 4
        final_cls = CapFC(in_cap_num=32*4*4, out_cap_num=10,
                          cap_dim=16, fc_manner='default')
    elif type == 'set1':
        module1 = CapConv2(ch_in=32*1, ch_out=32*2, groups=32,
                           residual=[True, True], iter_N=4, no_downsample=True,
                           more_skip=False,
                           layerwise_skip_connect=False,
                           wider_main_conv=False,
                           manner='0')
        module2 = CapConv2(ch_in=32*2, ch_out=32*4, groups=32,
                           residual=[True, True], iter_N=3,
                           more_skip=False,
                           layerwise_skip_connect=False,
                           wider_main_conv=False,
                           manner='0')
        module3 = CapConv2(ch_in=32*4, ch_out=32*8, groups=32,
                           residual=[True, True], iter_N=2,
                           more_skip=False,
                           layerwise_skip_connect=False,
                           wider_main_conv=False,
                           manner='3')
        module4 = CapConv2(ch_in=32*8, ch_out=32*16, groups=32,
                           residual=[True, True], iter_N=2,
                           more_skip=False,
                           layerwise_skip_connect=False,
                           wider_main_conv=False,
                           manner='3')
        # output after module4: bs, 32*16, 4, 4
        final_cls = CapFC(in_cap_num=32*4*4, out_cap_num=10,
                          cap_dim=16, fc_manner='default')
    elif type == 'set2':
        module1 = CapConv2(ch_in=32*1, ch_out=32*2, groups=32,
                           residual=[True, True], iter_N=4, no_downsample=True,
                           more_skip=True,
                           layerwise_skip_connect=True,
                           wider_main_conv=True,
                           manner='0')
        module2 = CapConv2(ch_in=32*2, ch_out=32*4, groups=32,
                           residual=[True, True], iter_N=4,
                           more_skip=True,
                           layerwise_skip_connect=True,
                           wider_main_conv=True,
                           manner='0')
        module3 = CapConv2(ch_in=32*4, ch_out=32*8, groups=32,
                           residual=[True, True], iter_N=3,
                           more_skip=True,
                           layerwise_skip_connect=False,
                           wider_main_conv=False,
                           manner='3')
        module4 = CapConv2(ch_in=32*8, ch_out=32*16, groups=32,
                           residual=[True, True], iter_N=2,
                           more_skip=True,
                           layerwise_skip_connect=False,
                           wider_main_conv=False,
                           manner='3')
        # output after module4: bs, 32*16, 4, 4
        final_cls = CapFC(in_cap_num=32*4*4, out_cap_num=10,
                          cap_dim=16, fc_manner='default')
    elif type == 'set3':
        "For now, wider and manner=3 are not compatible"
        module1 = CapConv2(ch_in=32*1, ch_out=32*2, groups=32,
                           residual=[True, True], iter_N=4, no_downsample=True,
                           more_skip=True,
                           layerwise_skip_connect=False,
                           wider_main_conv=True,
                           manner='0')
        module2 = CapConv2(ch_in=32*2, ch_out=32*4, groups=32,
                           residual=[True, True], iter_N=4,
                           more_skip=True,
                           layerwise_skip_connect=True,
                           wider_main_conv=False,
                           manner='3')
        module3 = CapConv2(ch_in=32*4, ch_out=32*8, groups=32,
                           residual=[True, True], iter_N=4,
                           more_skip=True,
                           layerwise_skip_connect=False,
                           wider_main_conv=True,
                           manner='0')
        module4 = CapConv2(ch_in=32*8, ch_out=32*16, groups=32,
                           residual=[True, True], iter_N=4,
                           more_skip=True,
                           layerwise_skip_connect=True,
                           wider_main_conv=False,
                           manner='3')
        # output after module4: bs, 32*16, 4, 4
        final_cls = CapFC(in_cap_num=32*4*4, out_cap_num=10,
                          cap_dim=16, fc_manner='default')
    elif type == 'set_OT':
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
        final_cls = CapFC(in_cap_num=32*4*4, out_cap_num=10,
                          cap_dim=16, fc_manner='default')
    elif type == 'set_OT_compare':
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
        module3 = CapConv2(ch_in=32*4, ch_out=32*8, groups=32,
                           residual=[True, True], iter_N=2,
                           layerwise_skip_connect=False,
                           wider_main_conv=False,
                           manner='0')
        module4 = CapConv2(ch_in=32*8, ch_out=32*16, groups=32,
                           residual=[True, True], iter_N=2,
                           layerwise_skip_connect=False,
                           wider_main_conv=False,
                           manner='0')
        # output after module4: bs, 32*16, 4, 4
        final_cls = CapFC(in_cap_num=32*4*4, out_cap_num=10,
                          cap_dim=16, fc_manner='default')

    elif type == 'set4':
        # based on set_OT
        group = 32

        module1 = CapConv2(ch_in=32*1, ch_out=32*2, groups=32,
                           residual=[True, True], iter_N=3, no_downsample=True,
                           layerwise_skip_connect=True,
                           wider_main_conv=False,
                           manner='4',
                           ot_choice='within')
        module1_ot_loss = OptTrans(ch_x=32*2, ch_y=32*2,
                                   spatial_x=32, spatial_y=32,
                                   group=group, C_form='cosine',
                                   no_bp_P_L=opts.no_bp_P_L)

        module2 = CapConv2(ch_in=32*2, ch_out=32*4, groups=32,
                           residual=[True, True], iter_N=3,
                           layerwise_skip_connect=False,
                           wider_main_conv=False,
                           manner='4',
                           ot_choice='within2')
        module2_ot_loss = OptTrans(ch_x=32*4, ch_y=32*2,
                                   spatial_x=16, spatial_y=32,
                                   group=group, C_form='cosine',
                                   no_bp_P_L=opts.no_bp_P_L)

        module3 = CapConv2(ch_in=32*4, ch_out=32*8, groups=32,
                           residual=[True, True], iter_N=3,
                           layerwise_skip_connect=True,
                           wider_main_conv=False,
                           manner='3',
                           ot_choice='within')
        module3_ot_loss = OptTrans(ch_x=32*8, ch_y=32*8,
                                   spatial_x=8, spatial_y=8,
                                   group=group, C_form='cosine',
                                   no_bp_P_L=opts.no_bp_P_L)

        module4 = CapConv2(ch_in=32*8, ch_out=32*16, groups=32,
                           residual=[True, True], iter_N=3,
                           layerwise_skip_connect=True,
                           wider_main_conv=False,
                           manner='3',
                           ot_choice='within2')
        module4_ot_loss = OptTrans(ch_x=32*16, ch_y=32*8,
                                   spatial_x=4, spatial_y=8,
                                   group=group, C_form='cosine',
                                   no_bp_P_L=opts.no_bp_P_L)

        # output after module4: bs, 32*16, 4, 4
        final_cls = CapFC(in_cap_num=32*4*4, out_cap_num=10,
                          cap_dim=16, fc_manner='default')

    elif type == 'resnet':
        assert (opts.depth - 2) % 6 == 0, 'depth should be 6n+2'
        inplanes = 16
        n = (opts.depth - 2) / 6
        block = Bottleneck if opts.depth >= 44 else BasicBlock

        if opts.dataset == 'tiny_imagenet':
            if opts.bigger_input:
                stride_num, fc_num, pool_stride, pool_kernel = 2, 64*4*4, 7, 7
            else:
                stride_num, fc_num, pool_stride, pool_kernel = 2, 64*2*2, 8, 8
        else:
            stride_num, fc_num, pool_stride, pool_kernel = 1, 64, 1, 8

        module1 = nn.Sequential(*[
            nn.Conv2d(input_ch, 16, kernel_size=3, padding=1, bias=False, stride=stride_num),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        ])

        module2 = _make_layer(block, 16, n)
        module3 = _make_layer(block, 32, n, stride=2)
        module4 = _make_layer(block, 64, n, stride=2)
        final_cls = nn.Sequential(*[
            nn.AvgPool2d(pool_kernel, stride=pool_stride),
            nn.Linear(fc_num, 10)
        ])

    return module1, module2, module3, module4, final_cls, \
           module1_ot_loss, module2_ot_loss, module3_ot_loss, module4_ot_loss
