from utils.from_wyang import AverageMeter, accuracy
from utils.util import *
from torch.optim.lr_scheduler import ReduceLROnPlateau, \
    ExponentialLR, MultiStepLR, StepLR, LambdaLR


def _update_all_data(all_data, stats):
    all_data[0].extend(stats[0])
    all_data[1].extend(stats[1])
    all_data[2].extend(stats[2])
    for i in range(21):
        all_data[3]['Y'][i].extend(stats[3]['Y'][i])
    return all_data


def set_lr_schedule(optimizer, plan, others=None):
    if plan == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, 'max',
                                      patience=25,
                                      factor=0.7,
                                      min_lr=0.00001)
    elif plan == 'multi_step':
        scheduler = MultiStepLR(optimizer,
                                milestones=others['milestones'],
                                gamma=others['gamma'])
    return scheduler


def adjust_learning_rate(optimizer, step, args):
    """
    Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Input: step/epoch
    """
    try:
        schedule_list = np.array(args.schedule)
    except AttributeError:
        schedule_list = np.array(args.schedule_cifar)
    decay = args.gamma ** (sum(step >= schedule_list))
    lr = args.lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def load_weights(model_path, model):
    print('test only mode, loading weights ...')
    checkpoints = torch.load(model_path)
    try:
        print('best test accu is: {:.4f}'.format(checkpoints['best_test_acc']))
    except KeyError:
        print('best test accu is: {:.4f}'.format(checkpoints['best_acc']))
    weights = checkpoints['state_dict']
    try:
        model.load_state_dict(weights)
    except KeyError:
        weights_new = collections.OrderedDict([(k[7:], v) for k, v in weights.items()])
        model.load_state_dict(weights_new)
    return model


def remove_batch(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            remove(os.path.join(dir, f))


def save_checkpoint(state, is_best, args, epoch):

    filepath = os.path.join(args.save_folder, 'epoch_{:d}.pth'.format(epoch+1))
    if (epoch+1) % args.save_epoch == 0 or epoch == 0:
        torch.save(state, filepath)
        print_log('model saved at {:s}'.format(filepath), args.file_name)
    if is_best:
        # save the best model
        remove_batch(args.save_folder, 'model_best')
        best_path = os.path.join(args.save_folder, 'model_best_at_epoch_{:d}.pth'.format(epoch+1))
        torch.save(state, best_path)
        print_log('best model saved at {:s}'.format(best_path), args.file_name)


def compute_KL(mean, std):
    loss = -0.5 * torch.sum(1 + torch.log(std**2) - mean**2 - std**2)
    return loss / std.size(0)


def train(trainloader, model, criterion, optimizer, opt, vis, epoch):

    use_cuda = opt.use_cuda
    structure = opt.model_cifar
    show_freq = opt.show_freq

    FIX_INPUT = False
    has_data = False
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    KL_losses = AverageMeter()
    normal_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # bar = Bar('Progressing', max=len(trainloader))
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
        if structure == 'capsule':
            outputs = outputs.norm(dim=2)

        # _, ind = outputs[4, :].max(0)
        # print('predict index: {:d}'.format(ind.data[0]))
        # one_sample = outputs[4, :]
        # check = torch.eq(one_sample, one_sample[0])
        # if check.sum().data[0] == len(one_sample):
        #     print('output is the same across all classes: {:.4f}\n'.format(one_sample[0].data[0]))
        if opt.use_spread_loss:
            loss = criterion(outputs, targets, epoch)
        else:
            loss = criterion(outputs, targets)
        if opt.use_KL:
            normal_losses.update(loss.data[0], inputs.size(0))
            loss_KL = opt.KL_factor * compute_KL(stats[-2], stats[-1])
            KL_losses.update(loss_KL.data[0], inputs.size(0))
            loss += loss_KL

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
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
                curr_info = {
                    'loss': losses.avg,
                    'KL_loss': KL_losses.avg,
                    'normal_loss': normal_losses.avg,
                    'acc': top1.avg,
                    'acc5': top5.avg,
                    'data': data_time.avg,
                    'batch': batch_time.avg,
                }
            else:
                curr_info = {
                    'loss': losses.avg,
                    'acc': top1.avg,
                    'acc5': top5.avg,
                    'data': data_time.avg,
                    'batch': batch_time.avg,
                }
            vis.print_loss(curr_info, epoch, batch_idx,
                           len(trainloader), epoch_sum=False, train=True)
            vis.plot_loss(errors=curr_info,
                          epoch=epoch, i=batch_idx, max_i=len(trainloader), train=True)
    return {
        'train_loss': losses.avg,
        'train_acc': top1.avg,
        'train_acc5': top5.avg,
    }


def test(testloader, model, criterion, opt, vis, epoch=0):

    use_cuda = opt.use_cuda
    structure = opt.model_cifar
    show_freq = opt.show_freq

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    stats_all_data = [[] for _ in range(4)]
    stats_all_data[3] = {'X': [], 'Y': [[] for _ in range(21)]}

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

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
            # The normal, rest testing procedure
            if structure == 'capsule':
                outputs = outputs.norm(dim=2)

            if opt.multi_crop_test:
                outputs = outputs.view(bs, ncrops, -1).mean(1)

            if opt.use_spread_loss:
                loss = criterion(outputs, targets, epoch)
            else:
                loss = criterion(outputs, targets)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data[0], inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % show_freq == 0 or batch_idx == len(testloader)-1:
                curr_info = {
                    'loss': losses.avg,
                    'acc': top1.avg,
                    'data': data_time.avg,
                    'batch': batch_time.avg,
                }
                vis.print_loss(curr_info, epoch, batch_idx,
                               len(testloader), epoch_sum=False, train=False)
                if opt.test_only is not True:
                    vis.plot_loss(errors=curr_info,
                                  epoch=epoch, i=batch_idx, max_i=len(testloader), train=False)
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
        'test_acc': top1.avg,
    }
