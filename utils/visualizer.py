from utils.utils import *
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')


class Visualizer(object):
    def __init__(self, opt, dataset=None):
        self.opt = opt
        if self.opt.no_visdom is False:
            import visdom
            name = opt.experiment_name
            self.vis = visdom.Visdom(port=opt.port_id, env=name)
            # loss/line 100, text 200, images/hist/etc 300
            self.dis_win_id_line = 100
            self.dis_win_id_txt = 200
            self.dis_win_id_im, self.dis_im_cnt, self.dis_im_cycle = 300, 0, 4
            self.loss_data = {'X': [], 'Y': [], 'legend': ['total_loss', 'loss_c', 'loss_l']}

    def plot_loss(self, errors, progress, others=None):
        """draw loss on visdom"""
        loss, loss_l, loss_c = errors[0].data[0], errors[1].data[0], errors[2].data[0]
        epoch, iter_ind, epoch_size = progress[0], progress[1], progress[2]
        x_progress = epoch + float(iter_ind/epoch_size)

        self.loss_data['X'].append([x_progress, x_progress, x_progress])
        self.loss_data['Y'].append([loss, loss_l, loss_c])

        self.vis.line(
            X=np.array(self.loss_data['X']),
            Y=np.array(self.loss_data['Y']),
            opts={
                'title': 'Train loss over time',
                'legend': self.loss_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.dis_win_id_line
        )

    def print_loss(self, info, progress, others=None):
        """show loss info in console"""
        loss, top1_acc, top5_acc = info[0], info[1], info[2]
        epoch, iter_ind, epoch_size = progress[0], progress[1], progress[2]
        t0, t1 = others[0], others[1]

        msg = '[{:s}]\tepoch/iter [{:d}/{:d}][{:d}/{:d}] ||\t' \
              'Loss: {:.4f}, top1_acc: {:.4f}, top5_acc: {:.4f} ||\t' \
              'Time: {:.4f} sec/image'.format(
                self.opt.experiment_name, epoch, self.opt.max_epoch-1, iter_ind, epoch_size-1,
                loss, top1_acc, top5_acc, (t1 - t0)/self.opt.batch_size_train)
        print_log(msg, self.opt.file_name)

    def print_info(self, progress, others):
        """print useful info on visdom"""
        epoch, iter_ind, epoch_size = progress[0], progress[1], progress[2]
        still_run, lr, time_per_iter = others[0], others[1], others[2]

        left_time = time_per_iter * (epoch_size-1-iter_ind + (self.opt.max_epoch-1-epoch)*epoch_size) / 3600 if \
            still_run else 0
        status = 'RUNNING' if still_run else 'DONE'
        dynamic = 'start epoch: {:d}, iter: {:d}<br/>' \
                  'curr lr {:.8f}<br/>' \
                  'progress epoch/iter [{:d}/{:d}][{:d}/{:d}]<br/><br/>' \
                  'est. left time: {:.4f} hours<br/>' \
                  'time/image: {:.4f} sec<br/>'.format(
                    self.opt.start_epoch, self.opt.start_iter,
                    lr,
                    epoch, self.opt.max_epoch, iter_ind, epoch_size,
                    left_time, time_per_iter/self.opt.batch_size)
        common_suffix = '<br/><br/>-----------<br/>batch_size: {:d}<br/>' \
                        'optim: {:s}<br/>'.format(
                            self.opt.batch_size, self.opt.optim)

        msg = 'phase: {:s}<br/>status: <b>{:s}</b><br/>'.format(self.opt.phase, status)\
              + dynamic + common_suffix
        self.vis.text(msg, win=self.dis_win_id_txt)

