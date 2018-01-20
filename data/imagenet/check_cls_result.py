# ref: https://github.com/pytorch/examples/blob/master/imagenet/main.py
from object_detection.utils.util import *
from object_detection.utils.eval import accuracy

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.parallel
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn

import numpy as np


def validate(val_loader, model, criterion):

    file_name = 'resent18_log.txt'
    cls_num = 1000
    acc_per_cls = {
        'top1': np.zeros([cls_num, 3], dtype=float),
        'top5': np.zeros([cls_num, 3], dtype=float),
    }
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        [prec1, prec5], acc_per_cls = \
            accuracy(output.data, target, topk=(1, 5), acc_per_cls=acc_per_cls)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 or i == len(val_loader):
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print_log('Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
              .format(top1=top1, top5=top5), file_name, init=True)
    for k, _ in acc_per_cls.items():
        acc_per_cls[k][:, 2] = acc_per_cls[k][:, 0] / acc_per_cls[k][:, 1]
    for i in range(cls_num):
        print_log('class {:4d}\t{:.4f}\t{:.4f}'
                  .format(i+1, acc_per_cls['top1'][i, 2], acc_per_cls['top5'][i, 2]), file_name)

    return top1.avg


cudnn.benchmark = True
model = models.resnet18(pretrained=True)
model = torch.nn.DataParallel(model).cuda()
criterion = nn.CrossEntropyLoss().cuda()

print_freq = 5
batch_size = 256
data_path = '/home/hongyang/dataset/imagenet_cls/cls'
valdir = os.path.join(data_path, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=True,
    num_workers=4, pin_memory=True)

validate(val_loader, model, criterion)


