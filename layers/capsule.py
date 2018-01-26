import torch.nn as nn
from layers.models.cifar.resnet import BasicBlock, Bottleneck
from layers.cap_layer import CapLayer, CapLayer2, CapFC, squash
from object_detection.utils.util import weights_init
import time
import torch
import numpy as np


class CapsNet(nn.Module):
    """
    Capsule network.
    """
    def __init__(self, opts, num_classes=100):
        super(CapsNet, self).__init__()

        self.use_imagenet = True if opts.dataset == 'tiny_imagenet' else False
        self.cap_model = opts.cap_model
        self.use_multiple = opts.use_multiple
        input_ch = 1 if opts.dataset == 'fmnist' else 3
        self.measure_time = opts.measure_time
        self.cap_N = opts.cap_N

        if self.cap_model == 'v_base':
            # baseline
            if hasattr(opts, 'depth'):
                depth = opts.depth
            else:
                depth = 20  # default value
            assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
            self.inplanes = 16
            n = (depth - 2) / 6
            block = Bottleneck if depth >= 44 else BasicBlock

            if opts.dataset == 'tiny_imagenet':
                if opts.bigger_input:
                    stride_num, fc_num, pool_stride, pool_kernel = 2, 64*4*4, 7, 7
                else:
                    stride_num, fc_num, pool_stride, pool_kernel = 2, 64*2*2, 8, 8
            else:
                stride_num, fc_num, pool_stride, pool_kernel = 1, 64, 1, 8

            self.conv1 = nn.Conv2d(input_ch, 16, kernel_size=3, padding=1, bias=False, stride=stride_num)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(block, 16, n)
            self.layer2 = self._make_layer(block, 32, n, stride=2)
            self.layer3 = self._make_layer(block, 64, n, stride=2)
            self.avgpool = nn.AvgPool2d(pool_kernel, stride=pool_stride)
            self.fc = nn.Linear(fc_num, num_classes)
        elif self.cap_model == 'v0':
            # update Jan 17: original capsule idea in the paper
            # first conv
            self.tranfer_conv = nn.Conv2d(input_ch, opts.pre_ch_num, kernel_size=9, padding=1, stride=2)  # 256x13x13
            self.tranfer_bn = nn.InstanceNorm2d(opts.pre_ch_num, affine=True) \
                if opts.use_instanceBN else nn.BatchNorm2d(opts.pre_ch_num)
            self.tranfer_relu = nn.ReLU(True)

            # second conv
            factor = 8 if opts.w_version is 'v2' else 1
            send_to_cap_ch_num = opts.primary_cap_num * factor
            self.tranfer_conv1 = nn.Conv2d(opts.pre_ch_num, send_to_cap_ch_num, kernel_size=3, stride=2)  # (say256)x6x6
            self.tranfer_bn1 = nn.InstanceNorm2d(send_to_cap_ch_num, affine=True) \
                if opts.use_instanceBN else nn.BatchNorm2d(send_to_cap_ch_num)
            self.tranfer_relu1 = nn.ReLU(True)

            # needed for large spatial input on imagenet
            if opts.bigger_input:
                self.max_pool = nn.MaxPool2d(9, stride=9)  # input 224, before cap, 54 x 54
            else:
                self.max_pool = nn.MaxPool2d(5, stride=5)  # input 128, before cap, 30 x 30

            # capsLayer
            self.cap_layer = CapLayer(opts, num_in_caps=opts.primary_cap_num*6*6, num_out_caps=num_classes,
                                      out_dim=16, num_shared=opts.primary_cap_num, in_dim=8)
        elif self.cap_model[0:2] == 'v1':

            self.layer1 = nn.Sequential(*[
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(True)
            ])  # 32 output
            self.cap1_conv = nn.Sequential(*[
                nn.Conv2d(32*1, 32*2, kernel_size=1, stride=1, groups=32),
                nn.BatchNorm2d(32*2),
                nn.ReLU(True)
            ])
            self.cap1_conv_sub = nn.Sequential(*[
                nn.Conv2d(32*2, 32*2, kernel_size=1, stride=1, groups=32),
                nn.BatchNorm2d(32*2),
                nn.ReLU(True)
            ])  # 32 output

            self.cap2_conv = nn.Sequential(*[
                nn.Conv2d(32*2, 32*4, kernel_size=3, stride=2, padding=1, groups=32),
                nn.BatchNorm2d(32*4),
                nn.ReLU(True)
            ])
            self.cap2_conv_sub = nn.Sequential(*[
                nn.Conv2d(32*4, 32*4, kernel_size=1, stride=1, groups=32),
                nn.BatchNorm2d(32*4),
                nn.ReLU(True)
            ])  # 16 output

            self.cap3_conv = nn.Sequential(*[
                nn.Conv2d(32*4, 32*8, kernel_size=3, stride=2, padding=1, groups=32),
                nn.BatchNorm2d(32*8),
                nn.ReLU(True)
            ])
            self.cap3_conv_sub = nn.Sequential(*[
                nn.Conv2d(32*8, 32*8, kernel_size=1, stride=1, groups=32),
                nn.BatchNorm2d(32*8),
                nn.ReLU(True)
            ])  # 8 output

            if self.cap_model == 'v1':
                self.cap_layer = CapLayer(
                    opts, num_in_caps=opts.primary_cap_num*8*8, num_out_caps=num_classes,
                    out_dim=16, in_dim=8, num_shared=opts.primary_cap_num)
            elif self.cap_model[0:3] == 'v1_':
                # increase cap_num and downsize spatial capsules
                self.cap4_conv1 = nn.Sequential(*[
                    nn.Conv2d(32*8, 32*16, kernel_size=3, stride=2, padding=1, groups=32),
                    nn.BatchNorm2d(32*16),
                    nn.ReLU(True)
                ])  # output: bs, 32*16, 4, 4

                if self.cap_model == 'v1_1':
                    # TODO: the following is just a mess-guess
                    self.cap4_conv2 = nn.Sequential(*[
                        nn.Conv2d(32*16, 5*16, kernel_size=1, stride=1),    # normal conv
                        nn.BatchNorm2d(5*16),
                        nn.ReLU(True)
                    ])
                    self.avgpool = nn.AvgPool2d((4, 2), stride=(1, 2))
                    self.avgpool2 = nn.AvgPool2d((2, 4), stride=(2, 1))

                elif self.cap_model == 'v1_3':
                    self.cap4_conv2 = nn.Sequential(*[
                        nn.Conv2d(32*16, 10*16, kernel_size=1, stride=1),   # normal conv
                        nn.BatchNorm2d(10*16),
                        nn.ReLU(True)
                    ])
                    self.avgpool = nn.AvgPool2d(4)

                elif self.cap_model == 'v1_4':
                    self.capFC = CapFC(in_cap_num=32*4*4, out_cap_num=num_classes, cap_dim=16)

        # init the network
        for m in self.modules():
            weights_init(m)

    def forward(self, x, target=None, curr_iter=0, vis=None):
        stats = []
        multi_cap_stats = []
        if self.measure_time:
            torch.cuda.synchronize()
            start = time.perf_counter()

        if self.cap_model == 'v_base':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)                  # 16 x 32 x 32
            x = self.layer2(x)                  # 32 x 16 x 16
            x = self.layer3(x)                  # 64(for depth=20) x 8 x 8
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        elif self.cap_model == 'v0':
            x = self.tranfer_conv(x)
            x = self.tranfer_bn(x)
            x = self.tranfer_relu(x)
            x = self.tranfer_conv1(x)
            x = self.tranfer_bn1(x)
            x = self.tranfer_relu1(x)
            if self.measure_time:
                torch.cuda.synchronize()
                print('the whole previous conv time: {:.4f}'.format(time.perf_counter() - start))
                start = time.perf_counter()
            if self.use_imagenet:
                x = self.max_pool(x)
            x, stats = self.cap_layer(x, target, curr_iter, vis)
            if self.measure_time:
                torch.cuda.synchronize()
                print('last cap total time: {:.4f}'.format(time.perf_counter() - start))

        elif self.cap_model[0:2] == 'v1':
            x = self.layer1(x)

            x = self.cap1_conv(x)
            x = self._do_squash2(x, num_shared=32)
            for i in range(self.cap_N):
                x = self.cap1_conv_sub(x)
                x = self._do_squash2(x, num_shared=32)

            x = self.cap2_conv(x)
            x = self._do_squash2(x, num_shared=32)
            for i in range(self.cap_N):
                x = self.cap2_conv_sub(x)
                x = self._do_squash2(x, num_shared=32)

            x = self.cap3_conv(x)
            x = self._do_squash2(x, num_shared=32)
            for i in range(self.cap_N):
                x = self.cap3_conv_sub(x)
                x = self._do_squash2(x, num_shared=32)

            if self.cap_model == 'v1':
                x, stats = self.cap_layer(x, target, curr_iter, vis)

            elif self.cap_model[0:3] == 'v1_':
                x = self.cap4_conv1(x)
                x = self._do_squash2(x, num_shared=32)

                if self.cap_model == 'v1_1':  # also being 'v1_2'
                    x = self.cap4_conv2(x)
                    x = self._do_squash2(x, num_shared=5)
                    if np.random.random(1) >= .5:
                        x = self.avgpool(x)
                    else:
                        x = self.avgpool2(x)
                    x = x.view(x.size(0), -1, 16)   # for cifar-10

                elif self.cap_model == 'v1_3':
                    x = self.cap4_conv2(x)    # bs, 160, 4x4
                    x = self.avgpool(x)
                    x = x.view(x.size(0), 10, 16)
                    x = squash(x)
                elif self.cap_model == 'v1_4':
                    x = x.view(x.size(0), 32, 16, x.size(2), x.size(3))
                    x = x.permute(0, 1, 3, 4, 2).contiguous()
                    x = x.view(x.size(0), -1, 16)
                    x = self.capFC(x)
                    x = squash(x)
        else:
            raise NameError('Unknown structure or capsule model type.')

        if self.use_multiple:
            stats = self._sort_up_multi_stats(multi_cap_stats)
        return x, stats

    def _sort_up_multi_stats(self, multi_cap_stats):
        stats = [multi_cap_stats[0][j] for j in range(len(multi_cap_stats[0]))]
        for i in range(1, len(multi_cap_stats)):
            for j in range(len(multi_cap_stats[0])):
                stats[j] = torch.cat((stats[j], multi_cap_stats[i][j]), dim=0)
        return stats

    def _do_squash(self, x):
        # do squash along the channel dimension
        spatial_size = x.size(2)
        input_channel = x.size(1)
        x = x.resize(x.size(0), x.size(1), int(spatial_size**2)).permute(0, 2, 1)
        x = squash(x)
        x = x.permute(0, 2, 1).resize(x.size(0), input_channel, spatial_size, spatial_size)
        return x

    def _do_squash2(self, x, num_shared):
        # do squash per capsule
        # x: bs, num_shared*cap_dim, spatial, spatial
        spatial_size = x.size(2)
        batch_size = x.size(0)
        cap_dim = int(x.size(1) / num_shared)
        # TODO: wrong
        x = x.view(batch_size, num_shared, cap_dim, spatial_size, spatial_size)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(batch_size, -1, cap_dim)
        x = squash(x)
        # revert back to orginal shape
        x = x.view(batch_size, num_shared, spatial_size, spatial_size, cap_dim)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(batch_size, -1, spatial_size, spatial_size)
        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, int(blocks)):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
