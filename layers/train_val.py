# from utils.from_wyang import AverageMeter, accuracy
from object_detection.utils.util import *
from object_detection.utils.eval import accuracy


def _update_all_data(all_data, stats):
    all_data[0].extend(stats[0])
    all_data[1].extend(stats[1])
    all_data[2].extend(stats[2])
    for i in range(21):
        all_data[3]['Y'][i].extend(stats[3]['Y'][i])
    return all_data


def compute_KL(mean, std):
    loss = -0.5 * torch.sum(1 + torch.log(std**2) - mean**2 - std**2)
    return loss / std.size(0)


def train(trainloader, model, criterion, optimizer, opt, vis, epoch):

    FIX_INPUT = False       # for quick debug

    model.train()
    has_data = False
    use_cuda = opt.use_cuda
    show_freq = opt.show_freq

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    KL_losses = AverageMeter()
    normal_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    epoch_size = len(trainloader)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if FIX_INPUT:
            if has_data:
                inputs, targets = fix_inputs, fix_targets
            else:
                fix_inputs, fix_targets = inputs, targets
                has_data = True
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        # update: last two entries have mean, std for KL loss
        outputs, stats = model(inputs, targets)  # 128 x 10 x 16
        try:
            outputs = outputs.norm(dim=2)
        except RuntimeError:
            outputs = outputs

        # _, ind = outputs[4, :].max(0)
        # print('predict index: {:d}'.format(ind.data[0]))
        # one_sample = outputs[4, :]
        # check = torch.eq(one_sample, one_sample[0])
        # if check.sum().data[0] == len(one_sample):
        #     print('output is the same across all classes: {:.4f}\n'.format(one_sample[0].data[0]))
        if opt.loss_form == 'spread':
            loss = criterion(outputs, targets, epoch)
        else:
            loss = criterion(outputs, targets)
        if opt.use_KL:
            normal_losses.update(loss.data[0], inputs.size(0))
            loss_KL = opt.KL_factor * compute_KL(stats[-2], stats[-1])
            KL_losses.update(loss_KL.data[0], inputs.size(0))
            loss += loss_KL

        # measure accuracy and record loss
        prec1, prec5, _ = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # OPTIMIZE
        start = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('iter bp time: {:.4f}\n'.format(time.time()-start))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % show_freq == 0 or batch_idx == len(trainloader)-1:
            if opt.use_KL:
                # TODO: merge the case w or w/o KL
                curr_info = {
                    'loss': losses.avg,
                    'KL_loss': KL_losses.avg,
                    'normal_loss': normal_losses.avg,
                    'acc': top1.avg,
                    'acc5': top5.avg,
                    'data': data_time.avg,
                    'batch': batch_time.avg,
                }

            vis.print_loss((losses.avg, top1.avg, top5.avg),
                           (epoch, batch_idx, epoch_size),
                           (data_time.avg, batch_time.avg))
            vis.plot_loss((losses.avg, top1.avg, top5.avg),
                          (epoch, batch_idx, epoch_size))
    return {
        # [dict], this is the result for one epoch
        'train_loss': losses.avg,
        'train_acc_error': 100. - top1.avg,
        'train_acc5_error': 100. - top5.avg,
    }


def test(testloader, model, criterion, opt, vis, epoch=0):

    use_cuda = opt.use_cuda
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    stats_all_data = [[] for _ in range(4)]
    stats_all_data[3] = {'X': [], 'Y': [[] for _ in range(21)]}

    print_log('Testing at epoch [{:d}/{:d}] ...'.format(epoch, opt.max_epoch))
    for batch_idx, (inputs, targets) in enumerate(testloader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), \
                          torch.autograd.Variable(targets)

        # SHOW histogram here
        if opt.draw_hist:
            if opt.which_batch_idx == batch_idx:
                input_vis = vis
            elif opt.which_batch_idx == -1:
                input_vis = vis   # draw all samples in the test
            else:
                input_vis = None
        else:
            input_vis = None

        # compute output
        if opt.multi_crop_test:
            bs, ncrops, c, h, w = inputs.size()
            inputs_ = inputs.view(-1, c, h, w)
        else:
            inputs_ = inputs
        # the computation of stats is in 'cap_layer.py'
        # 'stats' is the result of ONE mini-batch
        outputs, stats = model(inputs_, targets, batch_idx, input_vis)

        if input_vis is not None:
            if opt.which_batch_idx == -1:
                stats_all_data = _update_all_data(stats_all_data, stats)
            else:
                # TODO: for now if input_vis is True, no multi_crop_test is allowed
                for i in range(21):
                    stats[3]['Y'][i] = 0. \
                        if stats[3]['Y'][i] == [] else \
                        np.mean(stats[3]['Y'][i])
                plot_info = {
                    'd2_num': outputs.size(2),
                    'curr_iter': batch_idx,
                    'model': os.path.basename(opt.cifar_model),
                    'target': not opt.non_target_j
                }
                vis.plot_hist(stats, plot_info)

        if opt.draw_hist is False:
            # Do evaluation: the normal, rest testing procedure
            try:
                outputs = outputs.norm(dim=2)
            except RuntimeError:
                outputs = outputs

            if opt.multi_crop_test:
                outputs = outputs.view(bs, ncrops, -1).mean(1)

            if opt.loss_form == 'spread':
                loss = criterion(outputs, targets, epoch)
            else:
                loss = criterion(outputs, targets)
            # measure accuracy and record loss
            prec1, prec5, _ = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data[0], inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

    # draw stats for all data here
    if opt.draw_hist and opt.which_batch_idx == -1:
        for i in range(21):
            stats_all_data[3]['Y'][i] = 0. \
                if stats_all_data[3]['Y'][i] == [] else \
                np.mean(stats_all_data[3]['Y'][i])
        plot_info = {
            'model': os.path.basename(opt.cifar_model),
            'target': not opt.non_target_j
        }
        vis.plot_hist(stats_all_data, plot_info, all_sample=True)

    return {
        'test_loss': losses.avg,
        'test_acc_error': 100.0 - top1.avg,
        'test_acc5_error': 100.0 - top5.avg,
    }
