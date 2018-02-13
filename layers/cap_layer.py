import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import time
import numpy as np
import math
from layers.misc import compute_stats

# to find the different prediction during routing
FIND_DIFF = False
# to see the detailed values of b, c, v, delta_b
LOOK_INTO_DETAILS = False


class CapLayer(nn.Module):
    """
        The original capsule implementation in the NIPS paper;
        and detailed ablative analysis on it.
    """
    def __init__(self, opts, num_in_caps, num_out_caps,
                 out_dim, in_dim, num_shared):
        super(CapLayer, self).__init__()

        self.measure_time = opts.measure_time
        self.which_sample, self.which_j = 0, 0
        self.non_target_j = opts.non_target_j

        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num_shared = num_shared
        self.route_num = opts.route_num
        self.comp_cap = opts.comp_cap
        self.num_out_caps = num_out_caps
        self.num_in_caps = num_in_caps

        self.use_KL = opts.use_KL
        self.KL_manner = opts.KL_manner
        self.add_cap_BN_relu = opts.add_cap_BN_relu
        self.add_cap_dropout = opts.add_cap_dropout
        self.squash_manner = opts.squash_manner

        if self.comp_cap:
            self.fc1 = nn.Linear(num_in_caps, num_in_caps)
            self.relu1 = nn.ReLU(inplace=True)
            self.drop1 = nn.Dropout(p=.5)
            self.fc2 = nn.Linear(num_in_caps, num_out_caps)
            self.fc_time = opts.fc_time
        else:
            self.W = nn.Conv2d(num_shared*in_dim, num_shared*num_out_caps*out_dim,
                               kernel_size=1, stride=1, groups=num_shared)

        if self.add_cap_dropout:
            self.cap_droput = nn.Dropout2d(p=opts.dropout_p)

        if self.add_cap_BN_relu:
            # self.cap_BN = nn.BatchNorm2d(out_dim)
            self.cap_BN = nn.InstanceNorm2d(out_dim, affine=True)
            self.cap_relu = nn.ReLU(True)

    def forward(self, input, target=None, curr_iter=None, draw_hist=None):
        """
        target, curr_iter, draw_hist are for debugging or stats collection purpose only
        """

        start = []
        # for draw_hist (test)
        batch_cos_dist, batch_i_length, batch_cos_v, avg_len = [], [], [], []
        # for KL loss (train)
        mean, std = [], []
        bs, in_channels, h, w = input.size()

        # create b on the fly
        # if opts.b_init == 'rand':
        #     self.b = Variable(torch.rand(num_out_caps, num_in_caps), requires_grad=False)
        # elif opts.b_init == 'zero':
        #     self.b = Variable(torch.zeros(num_out_caps, num_in_caps), requires_grad=False)
        # elif opts.b_init == 'learn':
        #     self.b = Variable(torch.zeros(num_out_caps, num_in_caps), requires_grad=True)
        b = Variable(torch.zeros(bs, self.num_out_caps, self.num_in_caps),
                     requires_grad=False)

        if FIND_DIFF:
            pred_list = []
            pred_list.extend(list(range(bs)))
            pred_list.extend(target.data)

        if self.comp_cap:
            x = input.view(input.size(0), -1)
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.drop1(x)
            v = self.fc2(x)

        else:
            # 1. affine mapping (get 'pred')
            if self.measure_time:
                torch.cuda.synchronize()
                start = time.perf_counter()

            pred = self.W(input)
            # pred: bs, 5120, 6, 6
            # -> bs, 10, 16, 32x6x6 (view)
            spatial_size = pred.size(2)
            pred = pred.view(bs, self.num_shared, self.num_out_caps, self.out_dim,
                             spatial_size, spatial_size)
            pred = pred.permute(0, 2, 3, 1, 4, 5).contiguous()
            pred = pred.view(bs, pred.size(1), pred.size(2), -1)

            if self.add_cap_BN_relu:
                NotImplementedError()
                # pred = self.cap_BN(pred.permute(0, 3, 1, 2).contiguous())
                # pred = pred.permute(0, 2, 3, 1)
                # pred = self.cap_relu(pred)
            if self.add_cap_dropout:
                NotImplementedError()
                # v = self.cap_droput(v)

            if self.measure_time:
                torch.cuda.synchronize()
                print('\tcap W time: {:.4f}'.format(time.perf_counter() - start))

            # 2. dynamic routing (get 'v')
            start = time.perf_counter()
            for i in range(self.route_num):

                internal_start = time.perf_counter()
                c = F.softmax(b, dim=2)
                if self.measure_time:
                    torch.cuda.synchronize()
                    b_sftmax_t = time.perf_counter() - internal_start
                    t1 = time.perf_counter()

                s = torch.matmul(pred, c.unsqueeze(3)).squeeze()   # TODO: optimize time (0.0238)
                if self.measure_time:
                    torch.cuda.synchronize()
                    s_matmul_t = time.perf_counter() - t1
                    t2 = time.perf_counter()

                # shape of v: bs, out_cap_num, out_cap_dim
                v = squash(s, self.squash_manner)
                if self.measure_time:
                    torch.cuda.synchronize()
                    squash_t = time.perf_counter() - t2
                    t3 = time.perf_counter()

                # update b/c
                if i < self.route_num - 1:
                    if self.measure_time:
                        torch.cuda.synchronize()
                        permute_t = time.perf_counter() - t3
                        t4 = time.perf_counter()

                    # delta_b = torch.matmul(_pred, v.unsqueeze(3)).squeeze()  # TODO: super inefficient (0.1557s)
                    delta_b = torch.matmul(v.unsqueeze(2), pred).squeeze()
                    if self.measure_time:
                        torch.cuda.synchronize()
                        delta_matmul_t = time.perf_counter() - t4
                        t5 = time.perf_counter()

                    if FIND_DIFF:
                        v_all_classes = v.norm(dim=2)
                        _, curr_pred = torch.max(v_all_classes, 1)
                        pred_list.extend(curr_pred.data)

                    b = torch.add(b, delta_b)
                    if self.measure_time:
                        torch.cuda.synchronize()
                        add_t = time.perf_counter() - t5

                    if self.measure_time:
                        torch.cuda.synchronize()
                        torch.cuda.synchronize()
                        update_t = time.perf_counter() - t3
                else:
                    update_t, add_t, delta_matmul_t, permute_t = 0, 0, 0, 0

                if self.measure_time:
                    torch.cuda.synchronize()
                    print('\t\tRoute r={:d} time: {:.4f}'.format(i, time.perf_counter() - internal_start))
                    print('\t\t\tb_sftmax_t time: {:.4f}'.format(b_sftmax_t))
                    print('\t\t\ts_matmul_t time: {:.4f}'.format(s_matmul_t))
                    print('\t\t\tsquash_t time: {:.4f}'.format(squash_t))
                    print('\t\t\tupdate_t time: {:.4f}'.format(update_t))
                    print('\t\t\t\tpermute: {:.4f}'.format(permute_t))
                    print('\t\t\t\tdelta_matmul: {:.4f}'.format(delta_matmul_t))
                    print('\t\t\t\tadd: {:.4f}'.format(add_t))

            # routing ends
            if self.measure_time:
                torch.cuda.synchronize()
                print('\tcap Route (r={:d}) time: {:.4f}'.format(self.route_num, time.perf_counter() - start))

            if draw_hist:
                batch_cos_dist, batch_i_length, batch_cos_v, avg_len = \
                    compute_stats(target, pred, v, self.non_target_j)

            if self.use_KL:
                mean, std = compute_stats(None, pred, v, KL_manner=self.KL_manner)

            if FIND_DIFF:
                temp = np.asarray(pred_list)
                temp = np.resize(temp, (self.route_num+2, bs)).transpose()  # 128 x 5
                predict_ = temp[:, 2:]
                check_ = np.sum((predict_ - predict_[:, 0].reshape(bs, 1)), axis=1)
                diff_ind = np.nonzero(check_)[0]
                print('curr_iter {:d}:'.format(curr_iter))
                if diff_ind.shape == (0,):
                    HAS_DIFF = False
                    print('no difference prediction during routing!')
                else:
                    HAS_DIFF = True
                    print(temp[diff_ind, :])
                    print('\n')

        stats = [batch_cos_dist, batch_i_length, batch_cos_v, avg_len,
                 mean, std]
        return v, stats


class CapConv(nn.Module):
    def __init__(self, ch_num, groups,
                 N=1, ch_out=-1, manner='0',
                 kernel_size=1, stride=1, pad=0):
        super(CapConv, self).__init__()
        self.ch_num_in = ch_num
        self.ch_num_out = ch_num if ch_out == -1 else ch_out
        self.groups = groups
        self.iter_N = N

        if manner == '0':
            layers = []
            for i in range(self.iter_N):
                layers.append(nn.Conv2d(self.ch_num_in, self.ch_num_out,
                                        kernel_size=kernel_size, stride=stride,
                                        groups=self.groups, padding=pad))
                # TODO: change BN per capsule, along the channel
                layers.append(nn.BatchNorm2d(self.ch_num_out))
                layers.append(nn.ReLU(True))
                layers.append(conv_squash(self.groups))

        self.sub_layer = nn.Sequential(*layers)

    def forward(self, input):
        return self.sub_layer(input)


class CapFC(nn.Module):
    """
        given an input 4-D blob, generate the FC output;
        this layer assumes the input capsule's dim is the same as that of the output's capsule
    """
    def __init__(self, in_cap_num, out_cap_num, cap_dim, fc_manner='default'):
        super(CapFC, self).__init__()
        self.in_cap_num = in_cap_num
        self.out_cap_num = out_cap_num
        self.cap_dim = cap_dim
        self.fc_manner = fc_manner
        if fc_manner == 'default':
            self.weight = Parameter(torch.Tensor(cap_dim, in_cap_num, out_cap_num))
            self.reset_parameters()
        elif fc_manner == 'fc':
            self.fc_layer = nn.Linear(self.in_cap_num*self.cap_dim,
                                      self.out_cap_num*self.cap_dim)
        elif fc_manner == 'paper':
            self.W1 = Parameter(torch.Tensor(cap_dim, cap_dim*out_cap_num))
            self.W2 = Parameter(torch.Tensor(out_cap_num, in_cap_num, 1))
            self.reset_parameters()

    def forward(self, x):
        if self.fc_manner == 'default':
            x = x.view(x.size(0), -1, self.cap_dim, x.size(2), x.size(3))
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = x.view(x.size(0), -1, self.cap_dim)
            input = x.permute(0, 2, 1).contiguous().unsqueeze(2)
            input = torch.matmul(input, self.weight).squeeze().permute(0, 2, 1).contiguous()
        elif self.fc_manner == 'fc':
            x = x.view(x.size(0), -1)
            input = self.fc_layer(x)
            input = input.view(input.size(0), self.out_cap_num, self.cap_dim)
        elif self.fc_manner == 'paper':
            x = x.view(x.size(0), -1, self.cap_dim, x.size(2), x.size(3))
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = x.view(x.size(0), -1, self.cap_dim)
            input = torch.matmul(x, self.W1)
            input = input.view(input.size(0), self.in_cap_num, self.cap_dim, self.out_cap_num)
            input = input.permute(0, 3, 2, 1).contiguous()
            input = torch.matmul(input, self.W2).squeeze()
        return squash(input)

    def reset_parameters(self):
        if self.fc_manner == 'default':
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
        elif self.fc_manner == 'paper':
            stdv = 1. / math.sqrt(self.W1.size(1))
            self.W1.data.uniform_(-stdv, stdv)
            stdv = 1. / math.sqrt(self.W2.size(1))
            self.W2.data.uniform_(-stdv, stdv)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_cap_num=' + str(self.in_cap_num) \
            + ', out_cap_num=' + str(self.out_cap_num) \
            + ', cap_dim=' + str(self.cap_dim) \
            + ', fc_manner=' + self.fc_manner + ')'


class conv_squash(nn.Module):
    def __init__(self, num_shared):
        super(conv_squash, self).__init__()
        self.num_shared = num_shared

    def forward(self, x):
        spatial_size = x.size(2)
        batch_size = x.size(0)
        cap_dim = int(x.size(1) / self.num_shared)
        x = x.view(batch_size, self.num_shared, cap_dim, spatial_size, spatial_size)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(batch_size, -1, cap_dim)
        x = squash(x)
        # revert back to original shape
        x = x.view(batch_size, self.num_shared, spatial_size, spatial_size, cap_dim)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(batch_size, -1, spatial_size, spatial_size)
        return x


def squash(vec, manner='paper'):
    """given a 3-D (bs, num_cap, dim_cap) input, squash the capsules
    output has the same shape as input
    """
    EPS, coeff2, mean = 1e-20, [], []

    assert len(vec.size()) == 3
    norm = vec.norm(dim=2)

    if manner == 'paper':
        norm_squared = norm ** 2
        coeff = norm_squared / (1 + norm_squared)
        coeff2 = torch.unsqueeze((coeff/(norm+EPS)), dim=2)
        # coeff2: bs x num_cap x 1

    elif manner == 'sigmoid':
        try:
            mean = (norm.max() - norm.min()) / 2.0
        except RuntimeError:
            print(vec)
            print(norm)
        coeff = F.sigmoid(norm - mean)   # in-place bug
        coeff2 = torch.unsqueeze((coeff/norm), dim=2)

    return torch.mul(vec, coeff2)


class MarginLoss(nn.Module):

    def __init__(self, num_classes, pos=0.9, neg=0.1, lam=0.5):
        super(MarginLoss, self).__init__()
        self.num_classes = num_classes
        self.pos = pos
        self.neg = neg
        self.lam = lam

    def forward(self, output, target):
        # output, 128 x 10
        # print(output.size())
        gt = Variable(torch.zeros(output.size(0), self.num_classes), requires_grad=False)
        gt = gt.scatter_(1, target.unsqueeze(1), 1)
        zero = Variable(torch.zeros(1))
        pos_part = torch.max(zero, self.pos - output).pow(2)
        neg_part = torch.max(zero, output - self.neg).pow(2)
        loss = gt * pos_part + self.lam * (1-gt) * neg_part
        return loss.sum() / output.size(0)


class SpreadLoss(nn.Module):
    def __init__(self, opt, num_classes,
                 m_low=0.2, m_high=0.9, margin_epoch=20,
                 fix_m=False):
        super(SpreadLoss, self).__init__()
        self.num_classes = num_classes
        self.total_epoch = opt.max_epoch
        self.m_low = m_low
        self.m_high = m_high
        self.margin_epoch = margin_epoch
        self.interval = (m_high - m_low) / (self.total_epoch - 2*margin_epoch)
        self.fix_m = fix_m

    def forward(self, output, target, epoch):

        bs = output.size(0)
        if self.fix_m:
            m = self.m_high
        else:
            if epoch < self.margin_epoch:
                m = self.m_low
            elif epoch >= self.total_epoch - self.margin_epoch:
                m = self.m_high
            else:
                m = self.m_low + self.interval * (epoch - self.margin_epoch)

        target_output = torch.stack([output[i, target[i].data[0]] for i in range(bs)])
        loss = output - target_output + m
        gt = Variable(torch.zeros(output.size(0), self.num_classes), requires_grad=False)
        gt = gt.scatter_(1, target.unsqueeze(1), 1)
        loss = loss * (1-gt)
        loss = torch.clamp(loss, min=0).pow(2)   # 128 x 10
        return loss.sum() / output.size(0)
