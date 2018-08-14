from utils.utils import *


def train(trainloader, model, criterion, optimizer, opt, visual, epoch):

    model.train()
    show_freq = opt.show_freq

    batch_time, data_time = AverageMeter(), AverageMeter()
    losses = AverageMeter()
    top1, top5 = AverageMeter(), AverageMeter()
    end = time.time()
    epoch_size = len(trainloader)

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        data_time.update(time.time() - end)

        if opt.pt_new:
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        else:
            if opt.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs = torch.autograd.Variable(inputs)
            targets = torch.autograd.Variable(targets)

        outputs, _, _, ot_loss = model(inputs, targets)  # 128 x 10 x 16
        ot_flag = model.module.ot_loss if len(opt.device_id) > 1 else model.ot_loss
        try:
            outputs = outputs.norm(dim=2)
        except RuntimeError:
            outputs = outputs

        if opt.loss_form == 'spread':
            loss = criterion(outputs, targets, epoch)
        else:
            loss = criterion(outputs, targets)

        if ot_flag:
            curr_ot_loss = opt.ot_loss_fac * torch.sum(ot_loss)
            loss += curr_ot_loss

        loss *= opt.loss_fac
        # measure accuracy and record loss
        [prec1, prec5], _ = accuracy(outputs.data, targets.data, topk=(1, 5))
        if opt.pt_new:
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
        else:
            losses.update(loss.data[0], inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % show_freq == 0 or batch_idx == len(trainloader)-1:

            visual.print_loss((losses.avg, top1.avg, top5.avg),
                              (epoch, batch_idx, epoch_size),
                              (data_time.avg, batch_time.avg))
            if not opt.no_visdom:
                visual.plot_loss((losses.avg, top1.avg, top5.avg),
                                 (epoch, batch_idx, epoch_size))
    return {
        # [dict], this is the result for one epoch
        'train_loss': losses.avg,
        'train_acc_error': 100. - top1.avg,
        'train_acc5_error': 100. - top5.avg,
    }


def inference(model, inputs, targets, criterion, bs, ncrops, epoch, opt):
    outputs, _, _, _ = model(inputs, targets, phase='test')
    try:
        outputs = outputs.norm(dim=2)
    except RuntimeError:
        outputs = outputs

    if opt.multi_crop_test:
        outputs = outputs.view(bs, ncrops, -1).mean(1)

    if opt.loss_form == 'spread':
        test_loss = criterion(outputs, targets, epoch)
    else:
        test_loss = criterion(outputs, targets)
    return outputs, test_loss


def test(testloader, model, opt, visual, epoch=0, criterion=None):

    test_losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    model.eval()

    stats_all_data = [[] for _ in range(4)]
    stats_all_data[3] = {'X': [], 'Y': [[] for _ in range(21)]}

    print_log('\nTesting at epoch [{:d}/{:d}] ...'.format(epoch, opt.max_epoch-1))

    for batch_idx, (inputs, targets) in enumerate(testloader):

        if opt.pt_new:
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        else:
            if opt.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs = torch.autograd.Variable(inputs, volatile=True)
            targets = torch.autograd.Variable(targets)

        if opt.multi_crop_test:
            bs, ncrops, c, h, w = inputs.size()
            inputs_ = inputs.view(-1, c, h, w)
        else:
            inputs_ = inputs
            bs, ncrops = 0, 0

        if opt.pt_new:
            with torch.no_grad():
                outputs, test_loss = inference(model, inputs_, targets, criterion, bs, ncrops, epoch, opt)
        else:
            outputs, test_loss = inference(model, inputs_, targets, criterion, bs, ncrops, epoch, opt)

        # measure accuracy and record test loss
        [prec1, prec5], _ = accuracy(outputs.data, targets.data, topk=(1, 5))
        if opt.pt_new:
            test_losses.update(test_loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
        else:
            test_losses.update(test_loss.data[0], inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

    # ONE epoch ends
    return {
        'test_loss': test_losses.avg,
        'test_acc_error': 100.0 - top1.avg,
        'test_acc5_error': 100.0 - top5.avg,
    }


