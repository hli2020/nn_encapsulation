from layers.cap_layer import CapConv, CapConv2, CapFC, CapLayer
from layers.OT_module import OptTrans
from layers.misc import connect_list
import torch.nn as nn
from layers.models.cifar.resnet import BasicBlock, Bottleneck


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


def build_net(type, opts, num_classes):

    ot_loss = False
    input_ch = 1 if opts.dataset == 'fmnist' \
                    or opts.dataset == 'mnist' else 3

    module1, module2, module3, module4, final_cls = [], [], [], [], []
    module1_ot_loss, module2_ot_loss, module3_ot_loss, module4_ot_loss = [], [], [], []

    # DEFAULT SETTING FOR ENCAPNET
    if type == 'encapnet_default':
        # default; #64
        connect_detail = connect_list[opts.connect_detail]
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
        final_cls = CapFC(in_cap_num=32*4*4, out_cap_num=num_classes,
                          cap_dim=16, fc_manner=opts.fc_manner)

    elif type == 'encapnet_default_super':
        # default_super; #65
        module1 = CapConv2(ch_in=32*1, ch_out=32*2, groups=32,
                           residual=[True, True], iter_N=4, no_downsample=True,
                           layerwise_skip_connect=False,
                           wider_main_conv=False,
                           manner='3')
        module2 = CapConv2(ch_in=32*2, ch_out=32*4, groups=32,
                           residual=[True, True], iter_N=8,
                           layerwise_skip_connect=True,
                           wider_main_conv=False,
                           manner='3')
        module3 = CapConv2(ch_in=32*4, ch_out=32*8, groups=32,
                           residual=[True, True], iter_N=4,
                           layerwise_skip_connect=False,
                           wider_main_conv=False,
                           manner='3')
        module4 = CapConv2(ch_in=32*8, ch_out=32*16, groups=32,
                           residual=[True, True], iter_N=8,
                           layerwise_skip_connect=True,
                           wider_main_conv=False,
                           manner='3')
        # output after module4: bs, 32*16, 4, 4
        final_cls = CapFC(in_cap_num=32*4*4, out_cap_num=num_classes,
                          cap_dim=16, fc_manner='default')

    elif type == 'encapnet_set_OT':
        # set_OT; #67/70, etc.
        # only the last modules uses OT
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
        final_cls = CapFC(in_cap_num=32*4*4, out_cap_num=num_classes,
                          cap_dim=16, fc_manner='default')
    elif type == 'encapnet_set_OT_compare':
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
        final_cls = CapFC(in_cap_num=32*4*4, out_cap_num=num_classes,
                          cap_dim=16, fc_manner='default')
    elif type == 'encapnet_set4':
        # based on set_OT
        ot_loss = True
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
        final_cls = CapFC(in_cap_num=32*4*4, out_cap_num=num_classes,
                          cap_dim=16, fc_manner='default')

    # DEFAULT SETTING FOR RESNET
    elif type == 'resnet_default':
        assert (opts.depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (opts.depth - 2) / 6
        block = Bottleneck if opts.depth >= 44 else BasicBlock
        # if opts.dataset == 'tiny_imagenet':
        #     if opts.bigger_input:
        #         stride_num, fc_num, pool_stride, pool_kernel = 2, 64*4*4, 7, 7
        #     else:
        #         stride_num, fc_num, pool_stride, pool_kernel = 2, 64*2*2, 8, 8
        # else:
        stride_num, fc_num, pool_stride, pool_kernel = 1, 64, 1, 8

        inplanes = 16
        module1 = _make_layer(inplanes, block, 16, n)
        module2 = _make_layer(inplanes, block, 32, n, stride=2)
        module3 = _make_layer(inplanes, block, 64, n, stride=2)
        module4 = nn.AvgPool2d(pool_kernel, stride=pool_stride)
        final_cls = nn.Linear(fc_num, num_classes)

    elif type == 'resnet_same_shape':
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
        module4 = nn.Sequential(*[
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.AvgPool2d(8),
        ])  # 8 spatial output
        fc_num = 512
        final_cls = nn.Linear(fc_num, num_classes)

    elif type == 'resnet_similar_param':
        module1 = nn.Sequential(*[
            nn.Conv2d(32, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        ])  # 32 spatial output
        module2 = nn.Sequential(*[
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        ])  # 16 spatial output
        module3 = nn.Sequential(*[
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        ])  # 8 spatial output
        module4 = nn.Sequential(*[
            nn.Conv2d(512, 1300, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(1300),
            nn.ReLU(True),
            nn.AvgPool2d(8),
        ])  # 8 spatial output
        fc_num = 1300
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

        module4 = CapLayer(opts, num_in_caps=opts.primary_cap_num*8*8, in_dim=8,
                           num_out_caps=opts.primary_cap_num*8*8, out_dim=16,
                           num_shared=opts.primary_cap_num, route=opts.route,
                           as_conv_output=True)
        final_cls = CapLayer(opts, num_in_caps=opts.primary_cap_num*8*8, in_dim=16,
                             num_out_caps=num_classes, out_dim=16,
                             num_shared=opts.primary_cap_num, route=opts.route)

    elif type == 'encapnet_table1':
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
        ])  # 256 (32, 8), 8 x 8

        module4 = CapConv(ch_num=32*8, ch_out=32*16, groups=32,
                          kernel_size=(3,), stride=2, pad=(1,),
                          residual=True, manner=opts.manner)
        # output after module4: bs, 32*16, 4, 4
        final_cls = CapFC(in_cap_num=32*4*4, out_cap_num=num_classes,
                          cap_dim=16, fc_manner='default')

    return ot_loss, \
        module1, module2, module3, module4, final_cls, \
           module1_ot_loss, module2_ot_loss, module3_ot_loss, module4_ot_loss
