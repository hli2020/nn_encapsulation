from layers.cap_layer import CapConv2, CapFC


def build_net(type):
    module1, module2, module3, module4, final_cls = [], [], [], [], []

    if type == 'set1':
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

    return module1, module2, module3, module4, final_cls
