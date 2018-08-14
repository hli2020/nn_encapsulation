from __future__ import print_function
import torch
import re
import numpy as np
import os
import torch.nn as nn
import time
from torch.nn.modules.module import _addindent


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def remove(file_name):
    try:
        os.remove(file_name)
    except:
        pass


def print_log(msg, file=None, init=False, quiet_termi=False):

    if not quiet_termi:
        print(msg)
    if file is None:
        pass
    else:
        if init:
            remove(file)
        with open(file, 'a') as log_file:
            log_file.write('%s\n' % msg)


def show_jot_opt(opt):

    if opt.phase == 'train':
        file_name = os.path.join(opt.save_folder,
                                 'opt_{:s}_START_epoch_{:d}_iter_{:d}_END_{:d}.txt'.format(
                                     opt.phase, opt.start_epoch, opt.start_iter, opt.max_epoch))
    else:
        file_name = os.path.join(opt.save_folder, 'opt_{:s}.txt'.format(opt.phase))

    opt.file_name = file_name
    args = vars(opt)

    print_log('Experiment name: {:s}'.format(opt.experiment_name), file_name, init=True)
    print_log('------------ Train/test Options -------------', file_name)

    for k, v in sorted(args.items()):
        print_log('%s: %s' % (str(k), str(v)), file_name)
    print_log('------------------ End --------------------', file_name)
    return opt


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    params_num = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()

        if isinstance(modstr, str):
            modstr = _addindent(modstr, 2)
        elif isinstance(modstr, tuple):
            modstr = _addindent(modstr[0], 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        # rest = b if b > 0 else params
        # params_num = params_num + rest
        params_num += params

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr, params_num * 4. / (1024**2)


def adjust_learning_rate(optimizer, step, args):
    """
        Sets the learning rate to the initial LR decayed by gamma
        at every specified step/epoch

        Adapted from PyTorch Imagenet example:
        https://github.com/pytorch/examples/blob/master/imagenet/main.py

        step could also be epoch
    """
    schedule_list = np.array(args.schedule)
    decay = args.gamma ** (sum(step >= schedule_list))
    lr = args.lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def _remove_batch(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            remove(os.path.join(dir, f))


def save_checkpoint(state, is_best, args, epoch):

    filepath = os.path.join(args.save_folder, 'epoch_{:d}.pth'.format(epoch+1))
    test_acc_error = state['test_acc_err']
    best_test_acc_err = state['best_test_acc_err']

    if (epoch+1) % args.save_epoch == 0 \
            or epoch == 0 or (epoch+1) == args.max_epoch:
        torch.save(state, filepath)
        print_log('model (top1_err: {:.4f}) saved at {:s}'.format(
            test_acc_error, filepath), args.file_name)

    if is_best:
        # save the best model
        _remove_batch(args.save_folder, 'model_best')
        best_path = os.path.join(args.save_folder, 'model_best_at_epoch_{:d}.pth'.format(epoch+1))
        torch.save(state, best_path)
        print_log('best model (top1_err: {:.4f}) saved at {:s}'.format(
            best_test_acc_err, best_path), args.file_name)


def accuracy(output, target, topk=(1,), acc_per_cls=None):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    if acc_per_cls is not None:
        target_min, target_max = target.min(), target.max()
        for i in range(target_min, target_max + 1):
            acc_per_cls['top1'][i, 1] += sum(target == i)
            acc_per_cls['top5'][i, 1] += sum(target == i)

            _check = correct[:1].sum(dim=0)
            acc_per_cls['top1'][i, 0] += sum(_check[target == i])
            _check = correct[:5].sum(dim=0)
            acc_per_cls['top5'][i, 0] += sum(_check[target == i])

    return res, acc_per_cls


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def weights_init_cap(m):
    """init module weights"""
    if torch.__version__.startswith('0.4'):
        xavier = nn.init.xavier_normal_
        gaussian = nn.init.normal_
    elif torch.__version__.startswith('0.3'):
        xavier = nn.init.xavier_normal
        gaussian = nn.init.normal

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        xavier(m.weight.data)
        try:
            gaussian(m.bias.data)
        except:
            # some conv in resnet do not have bias
            pass
    elif isinstance(m, nn.BatchNorm2d) \
            or isinstance(m, nn.InstanceNorm2d) \
            or isinstance(m, nn.BatchNorm3d)\
            or isinstance(m, nn.InstanceNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        xavier(m.weight.data)
        gaussian(m.bias.data)