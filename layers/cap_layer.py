import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import time
import numpy as np
import math


def softmax_dim(input, axis=1):
    # will be DEPRECATED
    input_size = input.size()

    trans_input = input.transpose(axis, len(input_size)-1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    soft_max_2d = F.softmax(input_2d)  # TODO: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.

    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size)-1)


def squash(vec, manner='paper'):
    EPS = 1e-20
    # vec: bs, num_cap, dim_cap
    assert len(vec.size()) == 3
    # vec: 128 x 10 x 16
    norm = vec.norm(dim=2)
    if manner == 'paper':
        norm_squared = norm ** 2
        coeff = norm_squared / (1 + norm_squared)
        coeff2 = torch.unsqueeze((coeff/(norm+EPS)), dim=2)
    elif manner == 'sigmoid':
        try:
            mean = (norm.max() - norm.min()) / 2.0
        except RuntimeError:
            print(vec)
            print(norm)
        coeff = F.sigmoid(norm - mean)   # in-place bug
        coeff2 = torch.unsqueeze((coeff/norm), dim=2)
    # coeff2: 128 x 10 x 1
    return torch.mul(vec, coeff2)


def _update(x, y, a):
    for i in range(len(x)):
        a[int(x[i])].append(y[i])
    return a


def compute_stats(target, pred, v, non_target_j=False, KL_manner=-1):
    eps = 1e-12
    batch_cos_dist = []
    batch_i_length = []
    batch_cos_v = []
    avg_len = [[] for _ in range(21)]    # there are 21 bins

    bs = pred.size(0)
    num_i = pred.size(1)
    num_j = pred.size(2)
    d_j = pred.size(3)

    # THE FOLLOWING PROCESS IS ONE ITER WITHIN THE MINI-BATCH
    # for i in range(2): #range(bs):
    if KL_manner == 2:
        # use orientation of u_hat
        pred_mat_norm = pred / pred.norm(dim=3, keepdim=True)
    elif KL_manner == 1:
        # use cos_dist
        pred_mat_norm = pred / pred.norm(dim=3, keepdim=True)     # bs 1152 10 16
        pred_mat_norm = pred_mat_norm.permute(0, 2, 1, 3)   # bs 10 1152 16
        v_norm = v / v.norm(dim=2, keepdim=True)    # bs 10 16
        v_norm = v_norm.unsqueeze(dim=3)  # bs 10 16 1
        cos_v = torch.matmul(pred_mat_norm, v_norm)  # bs 10 1152 1

    else:
        # TODO: unoptimized version, compute per sample, for showing results.
        for i in range(bs):
            samplet_gt = (target[i].data[0]+1) % 10 if non_target_j else target[i].data[0]
            pred_mat_norm = pred[i, :, samplet_gt, :].squeeze() / \
                            (pred[i, :, samplet_gt, :].squeeze().norm(dim=1).unsqueeze(dim=1) + eps)   # 1152 x 16

            # # 1. cos_distance, i - i
            # cosine_dist = torch.matmul(pred_mat_norm, pred_mat_norm.t()).data
            # cosine_dist = cosine_dist.cpu().numpy()
            # new_data = []
            # for j in range(pred.size(1)):
            #     new_data.extend(cosine_dist[j, j:])
            # batch_cos_dist.extend(new_data)

            # 2. |u_hat|
            i_length = pred[i, :, samplet_gt, :].squeeze().norm(dim=1).data
            i_length.cpu().numpy()
            batch_i_length.extend(i_length)

            # 3. cos_dist, i - j
            v_norm = v[i, samplet_gt, :] / (v[i, samplet_gt, :].norm() + eps)
            v_norm = v_norm.unsqueeze(dim=1)  # 16 x 1
            cos_v = torch.matmul(pred_mat_norm, v_norm).squeeze().data
            cos_v = cos_v.cpu().numpy()
            batch_cos_v.extend(cos_v)

            # 4.1. avg_len
            x_list = np.floor(cos_v * 10 + 10)   # 1152
            y_list = pred[i, :, samplet_gt, :].squeeze().norm(dim=1).data.cpu().numpy()   # 1152
            avg_len = _update(x_list, y_list, avg_len)

        # # 4.2
        # avg_len_new = []
        # for i in range(21):
        #     avg_value = 0. if avg_len[i] == [] else np.mean(avg_len[i])
        #     avg_len_new.append(avg_value)
    if target is not None:
        return batch_cos_dist, batch_i_length, batch_cos_v, \
                {'X': list(range(21)), 'Y': avg_len}
    else:
        if KL_manner == 1:
            # previous std is: 128 x 10 x 1152 x 1, we should have sample-wise std and mean
            std = torch.std(cos_v.view(bs, -1), dim=1)      # 128 x 1
            mean = torch.mean(cos_v.view(bs, -1), dim=1)    # 128 x 1
        elif KL_manner == 2:
            std = torch.std(pred_mat_norm.view(-1, num_j*num_i, d_j), dim=1)     # 128 x 16
            mean = torch.mean(pred_mat_norm.view(-1, num_j*num_i, d_j), dim=1)   # 128 x 16
        return mean, std
# info = {
#     'sample_index': 4,
#     'j': samplet_gt,
#     'i_num': pred.size(1),
#     'd2_num': pred.size(3),
#     'curr_iter': curr_iter,
# }
# vis.plot_hist(cosine_dist, i_length, info)


class CapLayer(nn.Module):
    def __init__(self, opts, num_in_caps, num_out_caps,
                 out_dim, in_dim, num_shared):
        super(CapLayer, self).__init__()

        # for finding different prediction during routing
        self.FIND_DIFF = False
        self.CHECK_DETAILS = opts.look_into_details  # TODO here
        self.look_into_details = opts.look_into_details
        self.measure_time = opts.measure_time
        self.which_sample, self.which_j = 0, 0
        self.non_target_j = opts.non_target_j

        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num_shared = num_shared
        self.route_num = opts.route_num
        self.w_version = opts.w_version
        self.num_out_caps = num_out_caps
        self.num_in_caps = num_in_caps

        self.use_KL = opts.use_KL
        self.KL_manner = opts.KL_manner
        self.add_cap_BN_relu = opts.add_cap_BN_relu
        self.add_cap_dropout = opts.add_cap_dropout
        self.squash_manner = opts.squash_manner

        if opts.w_version == 'v2':
            # faster
            self.W = nn.Conv2d(num_shared*in_dim, num_shared*num_out_caps*out_dim,
                               kernel_size=1, stride=1, groups=num_shared)
        elif opts.w_version == 'v3':
            # for fair comparison
            if opts.fc_time > 0:    # for computing the parameter num
                self.fc1 = nn.Linear(num_in_caps, num_in_caps)
                self.relu1 = nn.ReLU(inplace=True)
                self.drop1 = nn.Dropout(p=.5)
            self.fc2 = nn.Linear(num_in_caps, num_out_caps)
            # self.do_squash = opts.do_squash  # DEPRECATED
            self.fc_time = opts.fc_time

        # if opts.b_init == 'rand':
        #     self.b = Variable(torch.rand(num_out_caps, num_in_caps), requires_grad=False)
        # elif opts.b_init == 'zero':
        #     self.b = Variable(torch.zeros(num_out_caps, num_in_caps), requires_grad=False)
        # elif opts.b_init == 'learn':
        #     self.b = Variable(torch.zeros(num_out_caps, num_in_caps), requires_grad=True)

        if self.add_cap_dropout:
            self.cap_droput = nn.Dropout2d(p=opts.dropout_p)
        if self.add_cap_BN_relu:
            # self.cap_BN = nn.BatchNorm2d(out_dim)  # loss is nan
            self.cap_BN = nn.InstanceNorm2d(out_dim, affine=True)
            self.cap_relu = nn.ReLU(True)

    def forward(self, input, target, curr_iter, vis):
        "target, curr_iter, vis are verbose"
        batch_cos_dist = []
        batch_i_length = []
        batch_cos_v = []
        avg_len = []
        mean, std = [], []   # for KL loss
        bs, in_channels, h, w = input.size()
        b = Variable(torch.zeros(bs, self.num_out_caps, self.num_in_caps),
                     requires_grad=False)

        if self.FIND_DIFF:
            pred_list = []
            pred_list.extend(list(range(bs)))
            pred_list.extend(target.data)

        if self.w_version == 'v2':

            # 1. pred
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

            # 2. routing
            start = time.perf_counter()
            for i in range(self.route_num):

                internal_start = time.perf_counter()
                c = F.softmax(b, dim=2)
                # old note (WRONG): c: 128(bs) x 10(j) x 1152(i), \sum_j = 1
                # c = softmax_dim(b, axis=1)
                if self.measure_time:
                    torch.cuda.synchronize()
                    b_sftmax_t = time.perf_counter() - internal_start
                    t1 = time.perf_counter()
                # print(pred.size())
                # print(c.size())
                # print('pred id')
                # print(pred.get_device())
                # print('c id')
                # print(c.get_device())

                s = torch.matmul(pred, c.unsqueeze(3)).squeeze()   # TODO: (0.0238)
                if self.measure_time:
                    torch.cuda.synchronize()
                    s_matmul_t = time.perf_counter() - t1
                    t2 = time.perf_counter()
                # Here 'squash' has an additional argument, 'squash_manner'
                # final output: v: bs x 10 x 16
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

                    # delta_b = torch.matmul(_pred, v.unsqueeze(3)).squeeze()  # TODO: super-time-consuming (0.1557s)
                    delta_b = torch.matmul(v.unsqueeze(2), pred).squeeze()
                    if self.measure_time:
                        torch.cuda.synchronize()
                        delta_matmul_t = time.perf_counter() - t4
                        t5 = time.perf_counter()
                    # if self.FIND_DIFF:
                    #     v_all_classes = v.norm(dim=2)
                    #     _, curr_pred = torch.max(v_all_classes, 1)
                    #     pred_list.extend(curr_pred.data)

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

            if vis is not None:
                batch_cos_dist, batch_i_length, batch_cos_v, avg_len = \
                    compute_stats(target, pred, v, self.non_target_j)
            if self.use_KL:
                mean, std = compute_stats(None, pred, v, KL_manner=self.KL_manner)

            if self.FIND_DIFF:
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

        elif self.w_version == 'v3':
            # COMPLETELY nothing to do with capsule
            x = input.view(input.size(0), -1)
            for i in range(self.fc_time):
                x = self.fc1(x)
                x = self.relu1(x)
                x = self.drop1(x)
            v = self.fc2(x)
            # if self.do_squash:
            #     v = squash(v)

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

        return v, [batch_cos_dist, batch_i_length,
                   batch_cos_v, avg_len, mean, std]


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


class CapFC(nn.Module):
    def __init__(self, in_cap_num, out_cap_num, cap_dim):
        super(CapFC, self).__init__()
        self.in_cap_num = in_cap_num
        self.out_cap_num = out_cap_num
        self.cap_dim = cap_dim
        self.weight = Parameter(torch.Tensor(cap_dim, in_cap_num, out_cap_num))
        self.reset_parameters()

    def forward(self, input):
        input = input.permute(0, 2, 1).contiguous().unsqueeze(2)
        output = torch.matmul(input, self.weight).squeeze().permute(0, 2, 1).contiguous()
        return output

    # TODO: figure out the init stuff
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_cap_num=' + str(self.in_cap_num) \
            + ', out_cap_num=' + str(self.out_cap_num) \
            + ', cap_dim=' + str(self.cap_dim) + ')'


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
