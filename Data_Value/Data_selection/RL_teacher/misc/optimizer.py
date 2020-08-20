import torch.optim as optim


def get_optimizer(configs):
    optimizer = configs['optimizer']
    base_lr = configs['base_lr']
    model = configs['model']
    built_in = {'SGD': optim.SGD, 'Adam': optim.Adam}
    momentum = configs.get('momentum', 0)
    if momentum != 0:
        return built_in[optimizer](model.parameters(), lr=base_lr, momentum=momentum, weight_decay=5e-4)
    else:
        return built_in[optimizer](model.parameters(), lr=base_lr, weight_decay=5e-4)
