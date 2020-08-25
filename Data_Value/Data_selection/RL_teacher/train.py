from __future__ import print_function, division, absolute_import

import argparse
from argparse import RawTextHelpFormatter

import torch

from core.teacher_student import TeacherStudentModel
from core.helper_functions import state_func

from misc.logger import create_logger

from hparams.register import get_hparams
from dataloader.dataloader import get_dataloader
from misc.lr_scheduler import get_scheduler
from misc.optimizer import get_optimizer

from torch.utils.tensorboard import SummaryWriter

# ================= define global parameters ===============
logger = None
saver = None
evaluator = None


def make_global_parameters(hparams):
    # logger_configs: output_path, cfg_name
    global logger
    logger_configs = hparams.logger_configs
    logger = create_logger(logger_configs['output_path'], logger_configs['cfg_name'])

    # global saver
    # saver_configs = hparams.saver_configs
    # saver = Saver(saver_configs['init_best_metric'], saver_configs['metric_name'], hparams, saver_configs['output_path'])


def main(hparams, run=None, gpu_num=0):

    device = torch.device('cuda:{}'.format(gpu_num) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(gpu_num)
    # teacher train set 45%
    # teacher dev set 5%
    # student train set 50%
    # student test set 100%
    # ==================================================================
    # dataloader:  teacher_train/student_train/dev/test
    # models: teacher_configs/student_configs
    # optional: max_t, tau, threshold, M, max_non_increasing_steps, num_classes
    global logger
    experiment_name = '_single_teacher'
    if run is not None:
        experiment_name += '_'+run
    writer = SummaryWriter(comment=experiment_name)
    # ==================== building data loader ========================
    _teacher_train_loader_configs = hparams.dataloader['teacher_train']
    _student_train_loader_configs = hparams.dataloader['student_train']
    _dev_loader_configs = hparams.dataloader['dev']
    _test_loader_configs = hparams.dataloader['test']

    teacher_train_loader = get_dataloader(_teacher_train_loader_configs)
    student_train_loader = get_dataloader(_student_train_loader_configs)
    dev_loader = get_dataloader(_dev_loader_configs)
    test_loader = get_dataloader(_test_loader_configs)

    # =================== building model ==============================
    _teacher_configs = hparams.models['teacher_configs']
    _student_configs = hparams.models['student_configs']
    _model_configs = {
        'student_configs': _student_configs,
        'teacher_configs': _teacher_configs
    }
    use_vae = hparams.models['teacher_configs'].get('use_vae', False)
    vae = None
    if use_vae:
        vae = VAE(device=device).to(device)
        vae.load_state_dict(torch.load('vae_model.pth'))
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
    model = TeacherStudentModel(_model_configs)
    model.train()
    model.cuda()
    #for name, params in model.named_parameters():
    #    print (name, params.size())
    # ================== set up lr scheduler=============================
    student_lr_scheduler = get_scheduler('student-cifar10')
    teacher_lr_scheduler = get_scheduler('teacher-cifar10')

    # ================== set up optimizer ===============================
    _student_optim_configs = hparams.optimizer['student_configs']
    _student_optim_configs['model'] = model.student_net
    _teacher_optim_configs = hparams.optimizer['teacher_configs']
    _teacher_optim_configs['model'] = model.teacher_net
    student_optimizer = get_optimizer(_student_optim_configs)
    teacher_optimizer = get_optimizer(_teacher_optim_configs)

    # ================= set up optional configs =========================
    max_t = hparams.optional.get('max_t', 24219)
    #max_t = 4000
    tau = hparams.optional.get('tau', 0.75) # 0.84
    #tau = 0.15
    threshold = hparams.optional.get('threshold', 0.5)
    M = hparams.optional.get('M', 48) # original = 128
    max_non_increasing_steps = hparams.optional.get('max_non_increasing_steps', 10)
    num_classes = hparams.optional.get('num_classes', 10)

    # =============== pack fit configs for fitting teacher model ========
    fit_configs = {
        'state_func': state_func,
        'dataloader':
            {
                'teacher': teacher_train_loader,
                'student': student_train_loader,
                'dev': dev_loader,
                'test': test_loader
            },
        'optimizer':
            {
                'teacher': teacher_optimizer,
                'student': student_optimizer
            },
        'lr_scheduler':
            {
                'teacher': teacher_lr_scheduler,
                'student': student_lr_scheduler
            },
        'logger': logger,
        'writer': writer,
        'max_t': max_t,
        'tau': tau,
        'threshold': threshold,
        'M': M,
        'max_non_increasing_steps': max_non_increasing_steps,
        'num_classes': num_classes,
        'use_vae': use_vae,
        'vae': vae
    }
    print ('Fitting the teacher starts.............')
    model.fit_teacher(fit_configs)
    contents = {
        'state_dict': model.state_dict(),
    }
    torch.save(contents, './model/b1-weight-decay-checkpoint-model.pth.tar')
    print ('Done. \nTesting the teaching policy............')
    curve = model.val_teacher(fit_configs)
    contents = {
        'state_dict': model.state_dict(),
        'curve': curve
    }
    torch.save(contents, './model/b1-weight-decay-checkpoint-with-curve.pth.tar')
    print ('Done')
    # saver.save(model, optimizer, latest_metric, epoch)


if __name__ == '__main__':
    # logger_configs: output_path, cfg_name
    # dataloader:  teacher_train/student_train/dev/test
    # models: teacher_configs/student_configs
    # optional: max_t, tau, threshold, M, max_non_increasing_steps, num_classes
    parser = argparse.ArgumentParser(description='Data selection using RL', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--hparams', default='cifar10_l2t', type=str, help='Choose hyper parameter configuration.\n[cifar10_l2t, multi_cifar10_l2t, cifar10_l2t_augment, cifar10_l2t_vae]')
    parser.add_argument('--run', type=str, help='experiment name')
    parser.add_argument('--gpu', type=int, help='gpu number', default=0)

    args = parser.parse_args()
    extra_info = None
    hparams = get_hparams(args.hparams)(extra_info)

    make_global_parameters(hparams)

    #print(hparams._items)
    main(hparams, args.run, args.gpu)

