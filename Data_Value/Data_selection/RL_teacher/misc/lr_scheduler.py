def student_learning_rate_scheduler(optimizer, iterations):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break
    if iterations == 32000:
        lr = lr * 0.1
        print ('Adjust learning rate to : ', lr)
    elif iterations == 48000:
        lr = lr * 0.1
        print ('Adjust learning rate to : ', lr)
    else:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def teacher_learning_rate_scheduler(optimizer, iterations):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break
    if iterations % 50 == 0:
        lr = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_scheduler(name):
    if name == 'student-cifar10':
        return student_learning_rate_scheduler
    elif name == 'teacher-cifar10':
        return teacher_learning_rate_scheduler
    else:
        raise NotImplementedError
