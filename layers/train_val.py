from utils.utils import *
from layers.misc import compute_KL, update_all_data


# To compute time, use profiler:
# http://pytorch.org/docs/master/autograd.html?highlight=profile#torch.autograd.profiler.profile
FIX_INPUT = False


def train(trainloader, model, criterion, optimizer, opt, visual, epoch):

    model.train()
    has_data = False
    use_cuda = opt.use_cuda
    show_freq = opt.show_freq
    FIX_INPUTS, FIX_TARGETS = [], []

    batch_time, data_time = AverageMeter(), AverageMeter()
    losses, KL_losses, normal_losses = AverageMeter(), AverageMeter(), AverageMeter()
    top1, top5 = AverageMeter(), AverageMeter()
    end = time.time()
    epoch_size = len(trainloader)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if FIX_INPUT:
            if has_data:
                inputs, targets = FIX_INPUTS, FIX_TARGETS
            else:
                FIX_INPUTS, FIX_TARGETS = inputs, targets
                has_data = True
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # Last two entries in 'stats' have mean, std for KL loss
        # 'activation' is for EM routing
        outputs, stats, activation, ot_info = \
            model(inputs, targets)  # 128 x 10 x 16
        ot_flag, ot_loss = ot_info[0], ot_info[1]
        try:
            outputs = outputs.norm(dim=2)
        except RuntimeError:
            outputs = outputs
        # outputs = activation  # way worse

        # print(outputs.size())
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
        # if opt.use_KL:
        #     normal_losses.update(loss.data[0], inputs.size(0))
        #     loss_KL = opt.KL_factor * compute_KL(stats[-2], stats[-1])
        #     KL_losses.update(loss_KL.data[0], inputs.size(0))
        #     loss += loss_KL
        if ot_flag:
            curr_ot_loss = opt.ot_loss_fac * torch.sum(ot_loss)
            # print('ot_loss is {:.4f}'.format(curr_ot_loss))
            loss += curr_ot_loss

        loss *= opt.loss_fac
        # measure accuracy and record loss
        [prec1, prec5], _ = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        # OPTIMIZE
        start = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print('after bp:')
        # w_grad_mean = torch.mean(model.cap_layer.W.weight.grad.data)
        # print('iter: {:d}, W_grad mean: {:.6f}'.format(
        #     batch_idx, w_grad_mean))
        # if np.isnan(w_grad_mean):
        #     a = 1
        # print('iter: {:d}, bias_grad mean: {:.6f}'.format(
        #     batch_idx, torch.mean(model.cap_layer.W.bias.grad.data)))

        # if opt.meaure_time:
        #     print('iter bp time: {:.4f}\n'.format(time.time()-start))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % show_freq == 0 or batch_idx == len(trainloader)-1:
            # if opt.use_KL:
            #     NotImplementedError()
            #     curr_info = {
            #         'loss': losses.avg,
            #         'KL_loss': KL_losses.avg,
            #         'normal_loss': normal_losses.avg,
            #         'acc': top1.avg,
            #         'acc5': top5.avg,
            #         'data': data_time.avg,
            #         'batch': batch_time.avg,
            #     }
            visual.print_loss((losses.avg, top1.avg, top5.avg),
                           (epoch, batch_idx, epoch_size),
                           (data_time.avg, batch_time.avg))
            visual.plot_loss((losses.avg, top1.avg, top5.avg),
                          (epoch, batch_idx, epoch_size))
    return {
        # [dict], this is the result for one epoch
        'train_loss': losses.avg,
        'train_acc_error': 100. - top1.avg,
        'train_acc5_error': 100. - top5.avg,
    }


def test(testloader, model, opt, visual, epoch=0, criterion=None):
    """criterion is verbose"""
    if opt.draw_hist is False:
        assert criterion is not None
    else:
        assert opt.multi_crop_test is False

    use_cuda = opt.use_cuda
    test_losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    model.eval()

    stats_all_data = [[] for _ in range(4)]
    stats_all_data[3] = {'X': [], 'Y': [[] for _ in range(21)]}

    print_log('Testing at epoch [{:d}/{:d}] ...'.format(epoch, opt.max_epoch))
    for batch_idx, (inputs, targets) in enumerate(testloader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = torch.autograd.Variable(inputs, volatile=True)
        targets = torch.autograd.Variable(targets)

        if opt.multi_crop_test:
            bs, ncrops, c, h, w = inputs.size()
            inputs_ = inputs.view(-1, c, h, w)
        else:
            inputs_ = inputs

        # the computation of stats is in 'cap_layer.py'
        # 'stats' is from the mini-batch in ONE iteration
        outputs, stats, _, _ = model(inputs_, targets,
                                     batch_idx, opt.draw_hist, phase='test')

        if opt.draw_hist:
            # skip test evaluation
            stats_all_data = update_all_data(stats_all_data, stats)
        else:
            # do evaluation as in the normal train/test procedure
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
            # measure accuracy and record test loss
            [prec1, prec5], _ = accuracy(outputs.data, targets.data, topk=(1, 5))
            test_losses.update(test_loss.data[0], inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

    # ONE epoch ends
    if opt.draw_hist:
        for i in range(21):
            stats_all_data[3]['Y'][i] = 0. \
                if stats_all_data[3]['Y'][i] == [] else \
                np.mean(stats_all_data[3]['Y'][i])
        plot_info = {
            'model': os.path.basename(opt.cifar_model),
            'target': not opt.non_target_j
        }
        visual.plot_hist(stats_all_data, plot_info, all_sample=True)

    return {
        'test_loss': test_losses.avg,
        'test_acc_error': 100.0 - top1.avg,
        'test_acc5_error': 100.0 - top5.avg,
    }
