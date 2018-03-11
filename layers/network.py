import torch.nn as nn
from layers.misc import connect_list
from layers.models.cifar.resnet import BasicBlock, Bottleneck
from layers.cap_layer import CapLayer, CapConv, CapConv2, CapFC
from layers.OT_module import OptTrans
from object_detection.utils.util import weights_init
from layers.models.net_config import build_net
from layers.misc import weights_init_cap
import time
import torch


class EncapNet(nn.Module):
    """
    Capsule network.
    """
    def __init__(self, opts, num_classes=100):
        super(EncapNet, self).__init__()

        self.use_imagenet = True if opts.dataset == 'tiny_imagenet' else False
        self.measure_time = opts.measure_time
        self.net_config = opts.net_config
        input_ch = 1 if opts.dataset == 'fmnist' or opts.dataset == 'mnist' else 3

        if self.net_config == 'resnet_default':
            self.module0 = nn.Sequential(*[
                nn.Conv2d(input_ch, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True)
            ])  # 32 spatial output
        else:
            self.module0 = nn.Sequential(*[
                nn.Conv2d(input_ch, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(True)
            ])  # 32 spatial output

        if self.net_config[0:6] == 'capnet':
            self.route = opts.route
            if opts.route == 'EM':
                self.generate_activate = nn.Sequential(*[
                    nn.Conv2d(256, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU()
                ])

        self.ot_loss, \
            self.module1, self.module2, self.module3, self.module4, self.final_cls, \
            self.module1_ot_loss, self.module2_ot_loss, self.module3_ot_loss, self.module4_ot_loss = \
            build_net(opts.net_config, opts, num_classes)

        # init the network
        for m in self.modules():
            weights_init_cap(m)

    def forward(self, x, target=None,
                curr_iter=0, vis=None, phase='train'):
        """final_activation is for EM routing; ot_loss applies to resnet and encapnet;
        phase is for OT_loss, don't compute it during test"""
        output, stats, final_activation = [], [], []
        if self.measure_time:
            torch.cuda.synchronize()
            start = time.perf_counter()

        x = self.module0(x)
        if self.ot_loss:
            # set ot_loss = [] (NOT 0) when multiple-gpu mode if you don't use ot_loss
            ot_loss = 0 if phase == 'train' else []

            x, out_list = self.module1(x)
            if phase == 'train':
                ot_loss += self.module1_ot_loss(out_list[1], out_list[0])

            x, out_list = self.module2(x)
            if phase == 'train':
                ot_loss += self.module2_ot_loss(out_list[1], out_list[0])

            x, out_list = self.module3(x)
            if phase == 'train':
                ot_loss += self.module3_ot_loss(out_list[1], out_list[0])

            x, out_list = self.module4(x)
            if phase == 'train':
                ot_loss += self.module4_ot_loss(out_list[1], out_list[0])
        else:
            ot_loss = []
            x = self.module1(x)
            x = self.module2(x)
            x = self.module3(x)

            # if self.measure_time:
            #     torch.cuda.synchronize()
            #     print('the whole previous conv time: {:.4f}'.format(time.perf_counter() - start))
            #     start = time.perf_counter()

            if self.net_config[0:6] == 'capnet':
                activate = self.generate_activate(x).view(x.size(0), -1) if self.route == 'EM' else None
                x, stats, activation = self.module4(x, target, curr_iter, vis, activate=activate)
            else:
                x = self.module4(x)

        if self.net_config[0:6] == 'resnet':
            x = x.view(x.size(0), -1)

        if self.net_config[0:6] == 'capnet':
            output, _, final_activation = self.final_cls(x, target, curr_iter, vis, activate=activation)
        else:
            output = self.final_cls(x)

        return output, stats, final_activation, [self.ot_loss, ot_loss]



