from utils.util import *
import numpy as np
import ntpath


class Visualizer(object):
    def __init__(self, opt):
        self.opt = opt
        if self.opt.visdom:
            import visdom
            if opt.draw_hist:
                target_suffix = 'non_target' if opt.non_target_j is True else 'target'
                data_scale = 'all_data' if opt.see_all_sample is True else 'part_data'
                name = opt.experiment_name + '_draw_hist' \
                       + '_' + target_suffix + '_' + data_scale
            else:
                name = opt.experiment_name
            self.vis = visdom.Visdom(port=opt.port, env=name)
            # lines 100, text 200, images/hist 300
            self.display_win_id = 100
            self.dis_win_id_im = 300

            legend = ['total_loss', 'KL_loss'] if self.opt.use_KL else ['train_loss']
            self.loss_data = {'X': [], 'Y': [], 'legend': legend}
            self.acc_data = {'X': [], 'Y': [], 'legend': ['train_acc', 'train_acc5']}
            self.loss_data_test = {'X': [], 'Y': [], 'legend': ['test_loss']}
            self.acc_data_test = {'X': [], 'Y': [], 'legend': ['test_acc']}

    def print_loss(self, errors, epoch, i=0, max_i=0, epoch_sum=False, train=False):

        if epoch_sum:
            print_log('Summary [{:s}]\tepoch [{:d}/{:d}]\t\t'
                      'train_loss: {:.5f}\ttest_loss: {:.5f}\ttrain_acc: {:.5f}\t'
                      'test_acc: {:.5f}\ttrain_acc5: {:.5f}\n'.format(
                        self.opt.experiment_name, epoch, self.opt.max_epoch,
                        errors['train_loss'], errors['test_loss'], errors['train_acc'],
                        errors['test_acc'], errors['train_acc5']), self.opt.file_name)
        else:
            prefix = 'Train' if train else 'Test'
            if train:
                if self.opt.use_KL:
                    print_log('{:s} [{:s}]\tepoch [{:d}/{:d}]|[{:d}/{:d}]\t'
                              'data: {:.3f}s |batch: {:.3f}s\t'
                              'loss: {:.5f}\t(KL: {:.5f} ||normal: {:.5f})\t'
                              'acc: {:.5f}\tacc5: {:.5f}'.format(
                                prefix, self.opt.experiment_name,
                                epoch, self.opt.max_epoch, i, max_i,
                                errors['data'], errors['batch'],
                                errors['loss'], errors['KL_loss'], errors['normal_loss'],
                                errors['acc'], errors['acc5']), self.opt.file_name)
                else:
                    print_log('{:s} [{:s}]\tepoch [{:d}/{:d}]\titer [{:d}/{:d}]\t\t'
                              'data: {:.3f}s | batch: {:.3f}s\t'
                              'loss: {:.5f}\tacc: {:.5f}\tacc5: {:.5f}'.format(
                                prefix, self.opt.experiment_name,
                                epoch, self.opt.max_epoch, i, max_i,
                                errors['data'], errors['batch'],
                                errors['loss'], errors['acc'], errors['acc5']), self.opt.file_name)
            else:
                # test phase
                print_log('{:s} [{:s}]\tepoch [{:d}/{:d}]\titer [{:d}/{:d}]\t\t'
                          'data: {:.3f}s | batch: {:.3f}s\t'
                          'loss {:.5f}\tacc: {:.5f}'.format(
                            prefix, self.opt.experiment_name,
                            epoch, self.opt.max_epoch, i, max_i,
                            errors['data'], errors['batch'],
                            errors['loss'], errors['acc']), self.opt.file_name)

    def plot_loss(self, errors, epoch, i, max_i, train=False):

        x_progress = epoch + float(i/max_i)

        if train:
            if self.opt.use_KL:
                self.loss_data['X'].append([x_progress, x_progress])
                self.loss_data['Y'].append([errors['loss'], errors['KL_loss']])
            else:
                self.loss_data['X'].append(x_progress)
                self.loss_data['Y'].append(errors['loss'])
            self.acc_data['X'].append([x_progress, x_progress])
            self.acc_data['Y'].append([errors['acc'], errors['acc5']])

            self.vis.line(
                X=np.array(self.loss_data['X']),
                Y=np.array(self.loss_data['Y']),
                opts={
                    'title': 'train loss over time',
                    'legend': self.loss_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_win_id
            )
            self.vis.line(
                X=np.array(self.acc_data['X']),
                Y=np.array(self.acc_data['Y']),
                opts={
                    'title': 'train accuracy over time',
                    'legend': self.acc_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'accuracy'},
                win=self.display_win_id + 1
            )
        else:
            self.loss_data_test['X'].append(x_progress)
            self.loss_data_test['Y'].append(errors['loss'])
            self.acc_data_test['X'].append(x_progress)
            self.acc_data_test['Y'].append(errors['acc'])
            self.vis.line(
                X=np.array(self.loss_data_test['X']),
                Y=np.array(self.loss_data_test['Y']),
                opts={
                    'title': 'test loss over time',
                    'legend': self.loss_data_test['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_win_id + 2
            )
            self.vis.line(
                X=np.array(self.acc_data_test['X']),
                Y=np.array(self.acc_data_test['Y']),
                opts={
                    'title': 'test accuracy over time',
                    'legend': self.acc_data_test['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'accuracy'},
                win=self.display_win_id + 3
            )

    def plot_hist(self, stats_data, info, all_sample=False):
        target_suffix = 'target' if info['target'] is True else 'non_target'
        title_suffix = 'batch_id={:d} Model: {:s}, {:s}'.format(
            info['curr_iter'],
            info['model'], target_suffix) \
            if all_sample is False else 'Model: {:s}, {:s}'.format(info['model'], target_suffix)
        data1 = stats_data[0]
        data2 = stats_data[1]
        data3 = stats_data[2]
        data4 = stats_data[3]

        # if all_sample is False:
        #     title_str = 'CosDist: i - i, ' + title_suffix
        #     self.vis.histogram(
        #         data1,
        #         win=self.dis_win_id_im,
        #         opts={
        #             'title': title_str,
        #             'xlabel': 'bin',
        #             'ylabel': 'percentage',
        #             'numbins': 60,
        #         },
        #     )
        #     self.dis_win_id_im += 1

        # title_str = '| u_hat_i |, ' + title_suffix
        # self.vis.histogram(
        #     data2,
        #     win=self.dis_win_id_im,
        #     opts={
        #         'title': title_str,
        #         'xlabel': 'bin',
        #         'ylabel': 'percentage',
        #         'numbins': 30
        #     },
        # )
        # self.dis_win_id_im += 1

        title_str = 'CosDist: i - j, ' + title_suffix
        self.vis.histogram(
            data3,
            win=self.dis_win_id_im,
            opts={
                'title': title_str,
                'xlabel': 'bin',
                'ylabel': 'percentage',
                'numbins': 30
            },
        )
        self.dis_win_id_im += 1

        title_str = 'AvgLen: i - j, ' + title_suffix
        self.vis.line(
            X=np.linspace(-1, 1, 21),
            Y=np.array(data4['Y']),
            win=self.dis_win_id_im,
            opts={
                'title': title_str,
                'xlabel': 'distance',
                'ylabel': 'length',
            },
        )
        self.dis_win_id_im += 1

    def show_image(self, epoch, images):
        # show in the visdom
        idx = 2 if self.opt.model == 'default' or self.opt.add_gan_loss else 1
        for label, image_numpy in images.items():
            self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                           win=self.display_win_id + idx)
            idx += 1

