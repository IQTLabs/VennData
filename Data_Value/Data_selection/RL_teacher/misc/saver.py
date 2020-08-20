import os
import torch
import shutil

__all__ = ['Saver']


class Saver(object):

    def __init__(self, init_best_metric, metric_name, hparams, output_path, save_every=10):
        self.best_metric = init_best_metric
        self.output_path = output_path
        self.metric_name = metric_name
        self.hparams = hparams
        self.save_every = save_every
        if not os.path.exists(output_path):
            os.mkdir(output_path)

    def save(self, model, optimizer, latest_metric, epoch, filename='checkpoint.pth.tar'):

        hparams = self.hparams
        output_path = self.output_path
        # filename = 'epoch_' + str(epoch) + '_' + filename
        if (epoch+1) % self.save_every == 0:
            filename = 'E-%d-'%(epoch + 1) + filename
        filename = os.path.join(output_path, filename)
        contents = {'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'metric': self.best_metric,
                    'latest_metric': latest_metric,
                    'hparams': hparams}

        torch.save(contents, filename)

        if latest_metric > self.best_metric:
            self.best_metric = latest_metric
            save_name = 'best_' + self.metric_name + '.pth.tar'
            save_name = os.path.join(output_path, save_name)
            shutil.copyfile(filename, save_name)
            return True
        return False
