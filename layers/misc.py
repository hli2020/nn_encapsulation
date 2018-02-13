import torch
import numpy as np


# residual connections for cap_model=v1_x
connect_list = {
    'only_sub':     [False, True, False, True, False, True, False],
    'all':          [True, True, True, True, True, True, True],
    'default':      [False, False, False, False, False, False, False],
}


def _update(x, y, a):
    for i in range(len(x)):
        a[int(x[i])].append(y[i])
    return a


def sort_up_multi_stats(multi_cap_stats):
    # DEPRECATED
    stats = [multi_cap_stats[0][j] for j in range(len(multi_cap_stats[0]))]
    for i in range(1, len(multi_cap_stats)):
        for j in range(len(multi_cap_stats[0])):
            stats[j] = torch.cat((stats[j], multi_cap_stats[i][j]), dim=0)
    return stats


def update_all_data(all_data, stats):
    """FOR draw_hist"""
    all_data[0].extend(stats[0])
    all_data[1].extend(stats[1])
    all_data[2].extend(stats[2])
    for i in range(21):
        all_data[3]['Y'][i].extend(stats[3]['Y'][i])
    return all_data


def compute_KL(mean, std):
    loss = -0.5 * torch.sum(1 + torch.log(std**2) - mean**2 - std**2)
    return loss / std.size(0)


def compute_stats(target, pred, v, non_target_j=False, KL_manner=-1):
    # TODO: unoptimized version, compute per sample
    """for KL (train) or for draw_hist (test)"""
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


