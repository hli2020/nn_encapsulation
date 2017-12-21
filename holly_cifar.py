from __future__ import print_function
import torch.optim as optim
import torch.utils.data as data
from data.create_dset import create_dataset
import torch.backends.cudnn as cudnn
import torch.nn as nn
from layers.modules.capsule import CapsNet
from layers.modules.cap_layer import MarginLoss, SpreadLoss
from layers.modules.cifar_train_val import *
from utils.visualizer import Visualizer
from utils.util import *
from option.train_opt import args   # for cifar we also has test here

args.show_freq = 5
args.show_test_after_epoch = -1
args = show_jot_opt(args)
vis = Visualizer(args)

test_loader = data.DataLoader(create_dataset(args, 'test'), args.test_batch,
                              num_workers=args.num_workers, shuffle=False)
train_loader = data.DataLoader(create_dataset(args, 'train'), args.train_batch,
                               num_workers=args.num_workers, shuffle=True)

model = CapsNet(num_classes=train_loader.dataset.num_classes, opts=args)
if args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay, momentum=args.momentum, )
elif args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay, betas=(args.beta1, 0.999))
elif args.optim == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay, momentum=args.momentum,
                              alpha=0.9, centered=True)

if args.scheduler is not None:
    scheduler = set_lr_schedule(optimizer, args.scheduler)
print_log(model, args.file_name)

# set loss
if args.model_cifar == 'capsule':
    if args.use_CE_loss:
        criterion = nn.CrossEntropyLoss()
    elif args.use_spread_loss:
        criterion = SpreadLoss(args, fix_m=args.fix_m,
                               num_classes=train_loader.dataset.num_classes)
    else:
        # default loss
        criterion = MarginLoss(num_classes=train_loader.dataset.num_classes)
elif args.model_cifar == 'resnet':
    criterion = nn.CrossEntropyLoss()

if args.use_cuda:
    criterion = criterion.cuda()
    # if args.deploy:
    # TODO: zombie process
    #     model = torch.nn.DataParallel(model).cuda()
    # else:
    model = model.cuda()
cudnn.benchmark = True

if args.test_only:
    test_model_list = [1, 20, 80, 200, 300]

    for _, i in enumerate(test_model_list):
        model_file = \
            os.path.join(args.save_folder, 'epoch_{:d}.pth'.format(i))
        print('loading weights of model [{:s}]'.format(os.path.basename(model_file)))
        model = load_weights(model_file, model)
        args.cifar_model = model_file
        info = test(test_loader, model, criterion, args, vis)
        print('test acc is {:.4f}'.format(info['test_acc']))
else:
    # train and test
    best_acc, best_epoch = 0, 0
    for epoch in range(args.epochs):

        old_lr = optimizer.param_groups[0]['lr']
        if epoch == args.start_epoch - 1:
            print_log('\ninit learning rate {:f} at iter {:d}\n'.format(
                old_lr, epoch), args.file_name)

        # TRAIN
        info = train(train_loader, model, criterion, optimizer, args, vis, epoch)
        # TEST
        if epoch > args.show_test_after_epoch:
            extra_info = test(test_loader, model, criterion, args, vis, epoch)
        else:
            extra_info = dict()
            extra_info['test_loss'], extra_info['test_acc'] = 0, 0

        # SHOW EPOCH SUMMARY
        info.update(extra_info)
        vis.print_loss(info, epoch, epoch_sum=True)

        # ADJUST LR
        if args.scheduler is not None:
            scheduler.step(extra_info['test_acc'])
        else:
            adjust_learning_rate(optimizer, epoch, args)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print_log('\nchange learning rate from {:f} to '
                      '{:f} at iter {:d}\n'.format(old_lr, new_lr, epoch), args.file_name)

        # SAVE model
        test_acc = 0 if extra_info['test_acc'] == 'n/a' else extra_info['test_acc']
        is_best = test_acc > best_acc
        best_epoch = epoch if is_best else best_epoch
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch':            epoch+1,
                'state_dict':       model.state_dict(),
                'test_acc':         test_acc,
                'best_test_acc':    best_acc,
                'optimizer':        optimizer.state_dict(),
        }, is_best, args, epoch)
        if args.use_KL:
            kl_info = '<br/>KL_factor: {:.4f}'.format(args.KL_factor)
        else:
            kl_info = ''
        common_suffix = 'curr best test acc {:.4f} at epoch {:d}<br/>' \
                        'curr lr {:f}<br/>' \
                        'epoch [{:d} | {:d}]{:s}'.format(
                            best_acc, best_epoch, new_lr, epoch,
                            args.epochs, kl_info)
        msg = 'status: <b>RUNNING</b><br/>' + common_suffix
        vis.vis.text(msg, win=200)

    print_log('Best acc: {:.4f}. Training done.'.format(best_acc), args.file_name)
    msg = 'status: <b>DONE</b><br/>' + common_suffix
    vis.vis.text(msg, win=200)







