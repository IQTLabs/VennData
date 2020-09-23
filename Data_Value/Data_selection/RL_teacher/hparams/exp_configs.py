from .hparams import HParams
from .register import register
from core.models.resnet import ResNet34, ResNet18, ResNet50
from core.helper_functions import evaluator
import torchvision.transforms as transforms
import pickle

#TODO add experiment config
seed = 666

@register("bionic_ac")
def bionic_ac(extra_info):
    global seed
    root = './'
    dataset = 'bionic_regroup'
    splits = ['teacher_train', 'student_train', 'dev', 'test']
    teacher_configs = {
        'input_dim': 25,
        'output_dim': 1,
        'use_vae': True,
        'policy': 'actor_critic'
    }
    student_configs = {
        'base_model': ResNet18(num_classes=2, linear_in=814592),
        'evaluator': evaluator
    }
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.RandomCrop((1200,1400)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataloader = {
        'teacher_train':
            {
                'dataset': dataset,
                'split': splits[0],
                'root': root,
                'transform': transform_train,
                'batch_size': 1,
                'shuffle': True
            },
        'student_train':
            {
                'dataset': dataset,
                'split': splits[1],
                'root': root,
                'transform': transform_train,
                'batch_size': 1,
                'shuffle': True
            },
        'dev':
            {
                'dataset': dataset,
                'split': splits[2],
                'root': root,
                'transform': transform_test,
                'batch_size': 1, #1250,
                'shuffle': False
            },
        'test':
            {
                'dataset': dataset,
                'split': splits[3],
                'root': root,
                'transform': transform_test,
                'batch_size': 1, #1250,
                'shuffle': False
            }
    }
    models = {
        'teacher_configs': teacher_configs,
        'student_configs': student_configs
    }

    optimizer = {
        'teacher_configs':
            {
                'base_lr': 0.02,
                'optimizer': 'SGD',
            },
        'student_configs':
            {
                'base_lr': 0.1,
                'optimizer': 'SGD',
                'momentum': 0.9
            }
    }

    optional = dict()
    logger_configs = {
        'output_path': './log',
        'cfg_name': '%s-actor-critic' % dataset
    }

    hparams = HParams(
        dataloader=dataloader,
        models=models,
        optimizer=optimizer,
        optional=optional,
        logger_configs=logger_configs,
        seed=seed
    )
    return hparams

@register("cifar10_ac")
def cifar10_ac(extra_info):
    global seed
    root = './'
    dataset = 'cifar10'
    splits = ['teacher_train', 'student_train', 'dev', 'test']
    teacher_configs = {
        'input_dim': 25,
        'output_dim': 1,
        'use_vae': True,
        'policy': 'actor_critic'
    }
    student_configs = {
        'base_model': ResNet34(),
        'evaluator': evaluator
    }
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataloader = {
        'teacher_train':
            {
                'dataset': dataset,
                'split': splits[0],
                'root': root,
                'transform': transform_train,
                'batch_size': 2,
                'shuffle': True
            },
        'student_train':
            {
                'dataset': dataset,
                'split': splits[1],
                'root': root,
                'transform': transform_train,
                'batch_size': 2,
                'shuffle': True
            },
        'dev':
            {
                'dataset': dataset,
                'split': splits[2],
                'root': root,
                'transform': transform_test,
                'batch_size': 200, #1250,
                'shuffle': False
            },
        'test':
            {
                'dataset': dataset,
                'split': splits[3],
                'root': root,
                'transform': transform_test,
                'batch_size': 200, #1250,
                'shuffle': False
            }
    }
    models = {
        'teacher_configs': teacher_configs,
        'student_configs': student_configs
    }

    optimizer = {
        'teacher_configs':
            {
                'base_lr': 0.02,
                'optimizer': 'SGD',
            },
        'student_configs':
            {
                'base_lr': 0.1,
                'optimizer': 'SGD',
                'momentum': 0.9
            }
    }

    optional = dict()
    logger_configs = {
        'output_path': './log',
        'cfg_name': '%s-actor-critic' % dataset
    }

    hparams = HParams(
        dataloader=dataloader,
        models=models,
        optimizer=optimizer,
        optional=optional,
        logger_configs=logger_configs,
        seed=seed
    )
    return hparams

@register("cifar10_l2t_augment")
def cifar10_l2t_augment(extra_info):
    global seed
    root = './'
    dataset = 'cifar10'
    splits = ['teacher_train', 'student_train', 'dev', 'test']
    teacher_configs = {
        'input_dim': 25,
        'output_dim': 5,
        'use_vae': False
    }
    student_configs = {
        'base_model': ResNet34(),
        'evaluator': evaluator
    }
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataloader = {
        'teacher_train':
            {
                'dataset': dataset,
                'split': splits[0],
                'root': root,
                'transform': transform_train,
                'batch_size': 2,
                'shuffle': True
            },
        'student_train':
            {
                'dataset': dataset,
                'split': splits[1],
                'root': root,
                'transform': transform_train,
                'batch_size': 2,
                'shuffle': True
            },
        'dev':
            {
                'dataset': dataset,
                'split': splits[2],
                'root': root,
                'transform': transform_test,
                'batch_size': 200, #1250,
                'shuffle': False
            },
        'test':
            {
                'dataset': dataset,
                'split': splits[3],
                'root': root,
                'transform': transform_test,
                'batch_size': 200, #1250,
                'shuffle': False
            }
    }
    models = {
        'teacher_configs': teacher_configs,
        'student_configs': student_configs
    }

    optimizer = {
        'teacher_configs':
            {
                'base_lr': 0.02,
                'optimizer': 'SGD',
            },
        'student_configs':
            {
                'base_lr': 0.1,
                'optimizer': 'SGD',
                'momentum': 0.9
            }
    }

    optional = {
            'M': 32
    }
    logger_configs = {
        'output_path': './log',
        'cfg_name': '%s-l2t-new-version-dgx1-batch-1-sgd-lr0.02' % dataset
    }

    hparams = HParams(
        dataloader=dataloader,
        models=models,
        optimizer=optimizer,
        optional=optional,
        logger_configs=logger_configs,
        seed=seed
    )
    return hparams

@register("cifar10_l2t_vae")
def cifar10_l2t_vae(extra_info):
    global seed
    root = './'
    dataset = 'cifar10'
    splits = ['teacher_train', 'student_train', 'dev', 'test']
    teacher_configs = {
        'input_dim': 25,
        'output_dim': 1,
        'use_vae': True
    }
    student_configs = {
        'base_model': ResNet34(),
        'evaluator': evaluator
    }
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataloader = {
        'teacher_train':
            {
                'dataset': dataset,
                'split': splits[0],
                'root': root,
                'transform': transform_train,
                'batch_size': 2,
                'shuffle': True
            },
        'student_train':
            {
                'dataset': dataset,
                'split': splits[1],
                'root': root,
                'transform': transform_train,
                'batch_size': 2,
                'shuffle': True
            },
        'dev':
            {
                'dataset': dataset,
                'split': splits[2],
                'root': root,
                'transform': transform_test,
                'batch_size': 200, #1250,
                'shuffle': False
            },
        'test':
            {
                'dataset': dataset,
                'split': splits[3],
                'root': root,
                'transform': transform_test,
                'batch_size': 200, #1250,
                'shuffle': False
            }
    }
    models = {
        'teacher_configs': teacher_configs,
        'student_configs': student_configs
    }

    optimizer = {
        'teacher_configs':
            {
                'base_lr': 0.02,
                'optimizer': 'SGD',
            },
        'student_configs':
            {
                'base_lr': 0.1,
                'optimizer': 'SGD',
                'momentum': 0.9
            }
    }

    optional = dict()
    logger_configs = {
        'output_path': './log',
        'cfg_name': '%s-l2t-new-version-dgx1-batch-1-sgd-lr0.02' % dataset
    }

    hparams = HParams(
        dataloader=dataloader,
        models=models,
        optimizer=optimizer,
        optional=optional,
        logger_configs=logger_configs,
        seed=seed
    )
    return hparams

@register("multi_cifar10_l2t")
def multi_cifar10_l2t(extra_info):
    global seed
    root = './'
    dataset = 'multi_cifar10'
    splits = ['teacher_train', 'student_train', 'dev', 'test']
    teacher_configs = {
        'input_dim': 25
    }
    student_configs = {
        'base_model': ResNet34(),
        'evaluator': evaluator
    }
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataloader = {
        'teacher_train':
            {
                'dataset': dataset,
                'split': splits[0],
                'root': root,
                'transform': transform_train,
                'batch_size': 2,
                'shuffle': True
            },
        'student_train':
            {
                'dataset': dataset,
                'split': splits[1],
                'root': root,
                'transform': transform_train,
                'batch_size': 2,
                'shuffle': True
            },
        'dev':
            {
                'dataset': dataset,
                'split': splits[2],
                'root': root,
                'transform': transform_test,
                'batch_size': 200, #1250,
                'shuffle': False
            },
        'test':
            {
                'dataset': dataset,
                'split': splits[3],
                'root': root,
                'transform': transform_test,
                'batch_size': 200, #1250,
                'shuffle': False
            }
    }
    models = {
        'teacher_configs': teacher_configs,
        'student_configs': student_configs
    }

    optimizer = {
        'teacher_configs':
            {
                'base_lr': 0.02,
                'optimizer': 'SGD',
            },
        'student_configs':
            {
                'base_lr': 0.1,
                'optimizer': 'SGD',
                'momentum': 0.9
            }
    }

    optional = dict()
    logger_configs = {
        'output_path': './log',
        'cfg_name': '%s-l2t-new-version-dgx1-batch-1-sgd-lr0.02' % dataset
    }

    hparams = HParams(
        dataloader=dataloader,
        models=models,
        optimizer=optimizer,
        optional=optional,
        logger_configs=logger_configs,
        seed=seed
    )
    return hparams

@register("cifar10_l2t")
def cifar10_l2t(extra_info):
    global seed
    root = './'
    dataset = 'cifar10'
    splits = ['teacher_train', 'student_train', 'dev', 'test']
    teacher_configs = {
        'input_dim': 25,
        'output_dim': 1
    }
    student_configs = {
        'base_model': ResNet34(),
        'evaluator': evaluator
    }
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataloader = {
        'teacher_train':
            {
                'dataset': dataset,
                'split': splits[0],
                'root': root,
                'transform': transform_train,
                'batch_size': 2,
                'shuffle': True
            },
        'student_train':
            {
                'dataset': dataset,
                'split': splits[1],
                'root': root,
                'transform': transform_train,
                'batch_size': 2,
                'shuffle': True
            },
        'dev':
            {
                'dataset': dataset,
                'split': splits[2],
                'root': root,
                'transform': transform_test,
                'batch_size': 200, #1250,
                'shuffle': False
            },
        'test':
            {
                'dataset': dataset,
                'split': splits[3],
                'root': root,
                'transform': transform_test,
                'batch_size': 200, #1250,
                'shuffle': False
            }
    }
    models = {
        'teacher_configs': teacher_configs,
        'student_configs': student_configs
    }

    optimizer = {
        'teacher_configs':
            {
                'base_lr': 0.02,
                'optimizer': 'SGD',
            },
        'student_configs':
            {
                'base_lr': 0.1,
                'optimizer': 'SGD',
                'momentum': 0.9
            }
    }

    optional = dict()
    logger_configs = {
        'output_path': './log',
        'cfg_name': '%s-l2t-new-version-dgx1-batch-1-sgd-lr0.02' % dataset
    }

    hparams = HParams(
        dataloader=dataloader,
        models=models,
        optimizer=optimizer,
        optional=optional,
        logger_configs=logger_configs,
        seed=seed
    )
    return hparams
