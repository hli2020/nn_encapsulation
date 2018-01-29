import torch.nn as nn
from layers.models.cifar.resnet import BasicBlock, Bottleneck
from layers.cap_layer import CapLayer, CapLayer2, squash
from object_detection.utils.util import weights_init
import time
import torch

# FOR debug, see the detailed values of b, c, v, delta_b
        # if self.FIND_DIFF and HAS_DIFF:
        #     self.which_sample = diff_ind[0]
        # if self.FIND_DIFF or self.look_into_details:
        #     print('sample index is: {:d}'.format(self.which_sample))
        # if target is not None:
        #     self.which_j = target[self.which_sample].data[0]
        #     if self.look_into_details:
        #         print('target is: {:d} (also which_j)'.format(self.which_j))
        # else:
        #     if self.look_into_details:
        #         print('no target input, just pick up a random j, which_j is: {:d}'.format(self.which_j))
        #
        # if self.look_into_details:
        #     print('u_hat:')
        #     print(pred[self.which_sample, :, self.which_j, :])
        # # start all over again
        # if self.look_into_details:
        #     b = Variable(torch.zeros(b.size()), requires_grad=False)
        #     for i in range(self.route_num):
        #
        #         c = softmax_dim(b, axis=1)              # 128 x 10 x 1152, c_nji, \sum_j = 1
        #         temp_ = [torch.matmul(c[:, zz, :].unsqueeze(dim=1), pred[:, :, zz, :].squeeze()).squeeze()
        #                  for zz in range(self.num_out_caps)]
        #         s = torch.stack(temp_, dim=1)
        #         v = squash(s, self.squash_manner)       # 128 x 10 x 16
        #         temp_ = [torch.matmul(v[:, zz, :].unsqueeze(dim=1), pred[:, :, zz, :].permute(0, 2, 1)).squeeze()
        #                  for zz in range(self.num_out_caps)]
        #         delta_b = torch.stack(temp_, dim=1).detach()
        #         if self.FIND_DIFF:
        #             v_all_classes = v.norm(dim=2)
        #             _, curr_pred = torch.max(v_all_classes, 1)
        #             pred_list.extend(curr_pred.data)
        #         b = torch.add(b, delta_b)
        #         print('[{:d}/{:d}] b:'.format(i, self.route_num))
        #         print(b[self.which_sample, self.which_j, :])
        #         print('[{:d}/{:d}] c:'.format(i, self.route_num))
        #         print(c[self.which_sample, self.which_j, :])
        #         print('[{:d}/{:d}] v:'.format(i, self.route_num))
        #         print(v[self.which_sample, self.which_j, :])
        #
        #         print('[{:d}/{:d}] v all classes:'.format(i, self.route_num))
        #         print(v[self.which_sample, :, :].norm(dim=1))
        #
        #         print('[{:d}/{:d}] delta_b:'.format(i, self.route_num))
        #         print(delta_b[self.which_sample, self.which_j, :])
        #         print('\n')
        # END of debug


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
        else:
            # different structures below
            ############ v1 ############
            self.buffer = nn.Sequential(*[
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(True)
            ])
            # then do squash in the forward pass
            # the new convolution capsule idea
            self.basic_cap = CapLayer2(opts, 64, 64, 8, 8, route_num=opts.route_num)
            self.cls_cap = CapLayer2(opts, 64, 64, 8, 10, as_final_output=True, route_num=opts.route_num)

            ############ v2,v3,v4,v5 ############
            # increase the spatial size x2 and channel number x1/2 (TOO SLOW)
            cap_dim = 128
            # self.buffer2 = nn.Sequential(*[
            #     nn.ConvTranspose2d(64, cap_dim, stride=2, kernel_size=1, output_padding=1),
            #     nn.ReLU(True)
            # ])
            self.buffer2 = nn.Sequential(*[
                nn.Conv2d(64, cap_dim, kernel_size=3, padding=1),
                nn.ReLU(True)
            ])
            self.cap_smaller_in_share = CapLayer2(
                opts, cap_dim, cap_dim, 8, 8, shared_size=4,
                route_num=opts.route_num)

            self.cap_smaller_in_out_share = CapLayer2(
                opts, cap_dim, cap_dim, 8, 8, shared_size=4,
                shared_group=2, route_num=opts.route_num)

            self.cls_smaller_in_share = CapLayer2(
                opts, cap_dim, cap_dim, 8, 10, shared_size=2,
                as_final_output=True, route_num=opts.route_num)

            # misc utilites and toys
            self.dropout = nn.Dropout2d(p=0.1)
            self.bummer = nn.Sequential(*[
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            ])

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

        elif self.cap_model == 'v1':
            x = self.buffer(x)
            x = self._do_squash(x)
            for i in range(self.cap_N):
                x, _ = self.basic_cap(x)
            x, stats = self.cls_cap(x)

        elif self.cap_model == 'v2':
            x = self.buffer2(x)
            x = self._do_squash(x)
            for i in range(self.cap_N):
                x, curr_stats = self.cap_smaller_in_share(x)
                multi_cap_stats.append(curr_stats)
            x, stats = self.cls_smaller_in_share(x)
            multi_cap_stats.append(stats)

        elif self.cap_model == 'v3':
            x = self.buffer2(x)
            x = self._do_squash(x)
            for i in range(self.cap_N):
                x, curr_stats = self.cap_smaller_in_out_share(x)
                multi_cap_stats.append(curr_stats)
            x, stats = self.cls_smaller_in_share(x)
            multi_cap_stats.append(stats)

        elif self.cap_model == 'v4_1':
            x = self.buffer2(x)
            x = self._do_squash(x)
            for i in range(self.cap_N):
                residual = x
                x, curr_stats = self.cap_smaller_in_share(x)
                multi_cap_stats.append(curr_stats)
                x += residual
            x, stats = self.cls_smaller_in_share(x)
            multi_cap_stats.append(stats)

        elif self.cap_model == 'v4_2':
            x = self.buffer2(x)
            x = self._do_squash(x)
            for i in range(self.cap_N):
                residual = x
                x = self.cap_smaller_in_share(x)
                x += residual
                x = self._do_squash(x)
            x = self.cls_smaller_in_share(x)

        elif self.cap_model == 'v4_3':
            x = self.buffer2(x)
            x = self._do_squash(x)
            for i in range(self.cap_N):
                residual = x
                x = self.cap_smaller_in_share(x)
                x = self.dropout(x)
                x += residual
            x = self.cls_smaller_in_share(x)

        elif self.cap_model == 'v4_4':
            x = self.buffer2(x)
            x = self._do_squash(x)
            for i in range(self.cap_N):
                residual = x
                x = self.cap_smaller_in_share(x)
                x = self.dropout(x)
                x += residual
                x = self._do_squash(x)
            x = self.cls_smaller_in_share(x)

        elif self.cap_model == 'v4_5':
            x = self.buffer2(x)
            x = self._do_squash(x)
            for i in range(self.cap_N):
                residual = x
                residual = self.dropout(residual)
                x = self.cap_smaller_in_share(x)
                x += residual
            x = self.cls_smaller_in_share(x)

        elif self.cap_model == 'v4_6':
            x = self.buffer2(x)
            x = self._do_squash(x)
            for i in range(self.cap_N):
                residual = x
                residual = self.dropout(residual)
                x = self.cap_smaller_in_share(x)
                x += residual
                x = self._do_squash(x)
            x = self.cls_smaller_in_share(x)

        elif self.cap_model == 'v4_7':
            x = self.buffer2(x)
            x = self._do_squash(x)
            for i in range(self.cap_N):
                residual = x
                residual = self.dropout(residual)
                x = self.cap_smaller_in_share(x)
                x = self.dropout(x)
                x += residual
            x = self.cls_smaller_in_share(x)

        elif self.cap_model == 'v4_8':
            x = self.buffer2(x)
            x = self._do_squash(x)
            for i in range(self.cap_N):
                residual = x
                residual = self.dropout(residual)
                x = self.cap_smaller_in_share(x)
                x = self.dropout(x)
                x += residual
                x = self._do_squash(x)
            x = self.cls_smaller_in_share(x)
        elif self.cap_model == 'v5_1':
            x = self.buffer2(x)
            x = self._do_squash(x)
            for i in range(self.cap_N):
                residual, x1, x2 = x, x, x
                x1, curr_stats = self.cap_smaller_in_share(x1)
                multi_cap_stats.append(curr_stats)
                x2, curr_stats = self.cap_smaller_in_out_share(x2)
                multi_cap_stats.append(curr_stats)
                x = residual + x1 + x2
            x, stats = self.cls_smaller_in_share(x)
            multi_cap_stats.append(stats)

        elif self.cap_model == 'v5_2':
            x = self.buffer2(x)
            x = self._do_squash(x)
            for i in range(self.cap_N):
                residual, x1, x2 = x, x, x
                x1, _ = self.cap_smaller_in_share(x1)
                x2, _ = self.cap_smaller_in_out_share(x2)
                x = residual + x1 + x2
                x = self._do_squash(x)
            x, stats = self.cls_smaller_in_share(x)
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


def softmax_dim(input, axis=1):
    # DEPRECATED
    input_size = input.size()

    trans_input = input.transpose(axis, len(input_size)-1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    # UserWarning: Implicit dimension choice for softmax has been deprecated.
    # Change the call to include dim=X as an argument.
    soft_max_2d = F.softmax(input_2d)

    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size)-1)


class CapLayer2(nn.Module):
    """
        Convolution Capsule Layer
        input:      [bs, in_dim (d1), spatial_size_1, spatial_size_1]
        output:     [bs, out_dim (d2), spatial_size_2, spatial_size_2]
                    or [bs, out_dim (d2), spatial_size_2] if as_final_output=True
        Args:
                    in_dim:             dim of input capsules, d1
                    out_dim:            dim of output capsules, d2
                    spatial_size_1:     spatial_size_1 ** 2 = num_in_caps, total # of input caps
                    spatial_size_2:     spatial_size_2 ** 2 = num_out_caps, total # of output caps
                    as_final_output:    if True, spatial_size_2 = num_out_caps
        Convolution parameters (W): nn.Conv2d(IN, OUT, kernel_size=1)
        Propagation coefficients (b or c): bs_j_i
    """
    def __init__(self,
                 opts, in_dim, out_dim, spatial_size_1, spatial_size_2,
                 route_num, as_final_output=False,
                 shared_size=-1, shared_group=1):
        super(CapLayer2, self).__init__()
        self.num_in_caps = int(spatial_size_1 ** 2)
        if as_final_output:
            self.num_out_caps = spatial_size_2
        else:
            self.num_out_caps = int(spatial_size_2 ** 2)
        self.use_KL = opts.use_KL
        self.KL_manner = opts.KL_manner
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.route_num = route_num
        self.as_final_output = as_final_output
        self.shared_size = shared_size
        self.shared_group = shared_group
        if shared_size == -1:
            self.num_conv_groups = 1
        else:
            assert spatial_size_1 % shared_size == 0
            self.num_conv_groups = int((spatial_size_1/shared_size) ** 2)

        if shared_group > 1:
            self.learnable_b = Variable(
                torch.normal(means=torch.zeros(shared_group), std=torch.ones(shared_group)),
                requires_grad=True)

        IN = int(self.in_dim * self.num_conv_groups)
        OUT = int(self.out_dim * self.num_out_caps * self.num_conv_groups / self.shared_group)
        self.W = nn.Conv2d(IN, OUT, groups=self.num_conv_groups, kernel_size=1, stride=1)

    def forward(self, x):
        # TODO: unoptimized
        mean, std = [], []   # for KL loss
        bs = x.size(0)
        # generate random b on-the-fly
        b = Variable(torch.rand(bs, self.num_out_caps, self.num_in_caps), requires_grad=False)

        start = time.time()
        # x: bs, d1, 32 (spatial_size_1), 32
        # -> W(x): bs, d2x16x16, 32 (spatial_size_1), 32
        if self.num_conv_groups != 1:
            # reshape the input x first
            x = x.resize(bs, self.in_dim * self.num_conv_groups, self.shared_size, self.shared_size)

        pred = self.W(x)
        pred = pred.resize(bs, int(self.num_out_caps / self.shared_group), self.out_dim, self.num_in_caps)

        if self.shared_group != 1:
            raw_pred = pred
            pred = raw_pred + self.learnable_b[0]
            for ind in range(1, self.shared_group):
                pred = torch.cat((pred, raw_pred + self.learnable_b[ind]), dim=1)
        # assert pred.size(1) == self.num_out_caps
        pred = pred.permute(0, 3, 1, 2).contiguous()
        # pred_i_j_d2
        # print('cap W time: {:.4f}'.format(time.time() - start))

        # routing starts
        start = time.time()
        for i in range(self.route_num):

            c = softmax_dim(b, axis=1)              # bs x j x i, c_nji, \sum_j = 1
            s = [torch.matmul(c[:, zz, :].unsqueeze(dim=1), pred[:, :, zz, :].squeeze()).squeeze()
                 for zz in range(self.num_out_caps)]
            s = torch.stack(s, dim=1)
            v = squash(s)                           # do squashing along the last dim, bs x j x d2
            delta_b = [torch.matmul(v[:, zz, :].unsqueeze(dim=1), pred[:, :, zz, :].permute(0, 2, 1)).squeeze()
                       for zz in range(self.num_out_caps)]
            delta_b = torch.stack(delta_b, dim=1).detach()
            b = torch.add(b, delta_b)
        # print('cap Route (r={:d}) time: {:.4f}'.format(self.route_num, time.time() - start))
        # routing ends
        # v: bs, num_out_caps, out_dim

        if self.use_KL:
            mean, std = compute_stats(None, pred, v, KL_manner=self.KL_manner)

        if not self.as_final_output:
            # v: eg., 64, 256(16x16), 20 -> 64, 20, 16, 16
            spatial_out = int(math.sqrt(self.num_out_caps))
            v = v.permute(0, 2, 1).resize(bs, self.out_dim, spatial_out, spatial_out)
        return v, [mean, std]