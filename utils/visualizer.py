from utils.utils import *
import numpy as np
from matplotlib import pyplot as plt
#import plotly.tools as tls
from scipy.misc import imread
plt.switch_backend('agg')


# this is a minor change for demo
# this is a further change for demo
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
            # for visualization
            # TODO: visualize in the training process
            self.num_classes = dataset.num_classes
            self.class_name = dataset.COCO_CLASSES_names
            self.color = plt.cm.hsv(np.linspace(0, 1, (self.num_classes-1))).tolist()
            # for both train and test
            self.save_det_res_path = os.path.join(self.opt.save_folder, 'det_result')
            mkdirs(self.save_det_res_path)

    def plot_loss(self, errors, progress, others=None):
        """draw loss on visdom console"""
        #TODO: set figure height and width in visdom
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

    def print_loss(self, errors, progress, others=None):
        """show loss info in console"""
        loss, loss_l, loss_c = errors[0].data[0], errors[1].data[0], errors[2].data[0]
        epoch, iter_ind, epoch_size = progress[0], progress[1], progress[2]
        t0, t1 = others[0], others[1]
        msg = '[{:s}]\tepoch/iter [{:d}/{:d}][{:d}/{:d}] ||\t' \
              'Loss: {:.4f}, loc: {:.4f}, cls: {:.4f} ||\t' \
              'Time: {:.4f} sec/image'.format(
                self.opt.experiment_name, epoch, self.opt.max_epoch, iter_ind, epoch_size,
                loss, loss_l, loss_c, (t1 - t0)/self.opt.batch_size)
        print_log(msg, self.opt.file_name)

    def print_info(self, progress, others):
        """print useful info on visdom console"""
        #TODO: test case and set size

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

    def show_image(self, progress, others=None):
        """for test, print log info in console and show detection results on visdom"""
        if self.opt.phase == 'test':
            name = os.path.basename(os.path.dirname(self.opt.det_file))
            i, total_im, test_time = progress[0], progress[1], progress[2]
            all_boxes, im, im_name = others[0], others[1], others[2]

            print_log('[{:s}][{:s}]\tim_detect:\t{:d}/{:d} {:.3f}s'.format(
                self.opt.experiment_name, name, i, total_im, test_time), self.opt.file_name)

            dets = np.asarray(all_boxes)
            result_im = self._show_detection_result(im, dets[:, i], im_name)
            result_im = np.moveaxis(result_im, 2, 0)
            win_id = self.dis_win_id_im + (self.dis_im_cnt % self.dis_im_cycle)
            self.vis.image(result_im, win=win_id,
                           opts={
                               'title': 'subfolder: {:s}, name: {:s}'.format(
                                   os.path.basename(self.opt.save_folder), im_name),
                               'height': 320,
                               'width': 400,
                           })
            self.dis_im_cnt += 1

    def _show_detection_result(self, im, results, im_name):

        plt.figure()
        plt.axis('off')     # TODO, still the axis remains
        plt.imshow(im)
        currentAxis = plt.gca()

        for cls_ind in range(1, len(results)):
            if results[cls_ind] == []:
                continue
            else:

                cls_name = self.class_name[cls_ind-1]
                cls_color = self.color[cls_ind-1]
                inst_num = results[cls_ind].shape[0]
                for inst_ind in range(inst_num):
                    if results[cls_ind][inst_ind, -1] >= self.opt.visualize_thres:

                        score = results[cls_ind][inst_ind, -1]
                        pt = results[cls_ind][inst_ind, 0:-1]
                        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                        display_txt = '{:s}: {:.2f}'.format(cls_name, score)

                        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=cls_color, linewidth=2))
                        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': cls_color, 'alpha': .5})
                    else:
                        break
        result_file = '{:s}/{:s}.png'.format(self.save_det_res_path, im_name[:-4])

        plt.savefig(result_file, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()
        # ref: https://github.com/facebookresearch/visdom/issues/119
        # plotly_fig = tls.mpl_to_plotly(fig)
        # self.vis._send({
        #     data=plotly_fig.data,
        #     layout=plotly_fig.layout,
        # })
        result_im = imread(result_file)
        return result_im

# idx = 2 if self.opt.model == 'default' or self.opt.add_gan_loss else 1
# for label, image_numpy in images.items():
#     self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
#                    win=self.display_win_id + idx)
#     idx += 1
# if args.use_visdom:
#     random_batch_index = np.random.randint(images.size(0))
#     args.vis.image(images.data[random_batch_index].cpu().numpy())



