from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import math
import torch.nn as nn
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms.functional as TF
import random

from core.student_network import StudentNetwork
from core.teacher_network import TeacherNetwork,TeacherNetworkExtended

from misc.utils import init_params, to_var

# ================== helper function ===============
def to_generator(data):
    yield data

# ==================================================


class TeacherStudentModel(nn.Module):

    def __init__(self, configs):
        super(TeacherStudentModel, self).__init__()
        self.configs = configs

        self.student_net = StudentNetwork(configs['student_configs'])
        init_params(self.student_net)

        self.is_vae_teacher = self.configs['teacher_configs'].get('use_vae', False)
        #self.is_vae_teacher = self.configs['use_vae']
        if self.is_vae_teacher:
            self.teacher_net = TeacherNetworkExtended(configs['teacher_configs'])
        else:
            self.teacher_net = TeacherNetwork(configs['teacher_configs'])
        self.is_augment_teacher = self.teacher_net.output_dim > 1


    def fit_teacher(self, configs):
        '''
        :param configs:
            Required:
                state_func: [function] used to compute the state vector

                dataloader: [dict]
                    teacher: teacher training data loader
                    student: student training data loader
                    dev: for testing the student model so as to compute reward for the teacher
                    test: student testing data loader

                optimizer: [dict]
                    teacher: the optimizer for teacher
                    student: the optimizer for student

                lr_scheduler: [dict]
                    teahcer: the learning rate scheduler for the teacher model
                    student: the learning rate scheduler for the student model

                <del>current_epoch: [int] the current epoch</del>
                <del>total_epochs: the max number of epochs to train the model</del>
                logger: the logger

            Optional:
                max_t: [int] [50,000]
                    the maximum number iterations before stopping the teaching
                    , and once reach this number, return a reward 0.
                tau: [float32] [0.8]
                    the expected accuracy of the student model on dev set
                threshold: [float32] [0.5]
                    the probability threshold for choosing a sample.
                M: [int] [128]
                    the required batch-size for training the student model.
                max_non_increasing_steps: [int] [10]
                    The maximum number of iterations of the reward not increasing.
                    If exceeds it, stop training the teacher model.
                num_classes: [int] [10]
                    the number of classes in the training set.
        :return:
        '''
        teacher = self.teacher_net
        student = self.student_net
        # ==================== fetch configs [optional] ===============
        max_t = configs['max_t']
        tau = configs['tau']
        M = configs['M']
        max_non_increasing_steps = configs['max_non_increasing_steps']
        num_classes = configs['num_classes']

        # =================== fetch configs [required] ================
        state_func = configs['state_func']
        teacher_dataloader = configs['dataloader']['teacher']
        dev_dataloader = configs['dataloader']['dev']
        teacher_optimizer = configs['optimizer']['teacher']
        student_optimizer = configs['optimizer']['student']
        teacher_lr_scheduler = configs['lr_scheduler']['teacher']
        student_lr_scheduler = configs['lr_scheduler']['student']
        logger = configs['logger']
        writer = configs['writer']
        vae = configs['vae']

        # ================== init tracking history ====================
        rewards = []
        training_loss_history = []
        val_loss_history = []
        num_steps_to_achieve = []

        non_increasing_steps = 0
        student_updates = 0
        teacher_updates = 0
        best_acc_on_dev = 0
        effective_train_data = 0
        i_teacher = 0

        while True:
            i_tau = 0
            actions = []
            action_counts = np.array([])

            def overloaded_init_params(x):
                init_params(x)
                # if pointer == 0:
                #    init_params(x)
                # else:
                #     file_name = './model/resnet34-%5.4f.pth.tar' % (tau_list[pointer - 1])
                #     logger.info('Loaded model from' + file_name)
                #     x.load_state_dict(torch.load(file_name)['state_dict'])
            #label_counts = None
            label_counts = np.array([])
            while i_tau < max_t:
                i_teacher += 1
                i_tau += 1
                count = 0
                input_pool = []
                label_pool = []
                batch_actions = []
                # ================== collect training batch ============
                while True:
                    for idx, (inputs, labels) in enumerate(teacher_dataloader):
                        inputs = to_var(inputs)
                        labels = to_var(labels)
                        state_configs = {
                            'num_classes': num_classes,
                            'labels': labels,
                            'inputs': inputs,
                            'student': student.train(),
                            'current_iter': i_tau,
                            'max_iter': max_t,
                            'train_loss_history': training_loss_history,
                            'val_loss_history': val_loss_history,
                            'use_vae': self.is_vae_teacher,
                            'vae': vae
                        }
                        states, vae_z = state_func(state_configs)
                        _inputs = {'input': states.detach(), 'vae_z': vae_z}
                        predicts = teacher(_inputs)
                        if self.is_augment_teacher: # data augment teacher
                            m = torch.distributions.categorical.Categorical(predicts)
                            sampled_actions = m.sample() # n values (in the range of 0:action_space)
                            action_counts = np.append(action_counts, sampled_actions.detach().cpu().numpy())
                            # (0=>no select,1=>select w/ no aug, 2=>, horizontal flip, 3=> adjust brightness, 4=> adjust contrast)
                            action_log_probs = m.log_prob(sampled_actions)
                        else:
                            sampled_actions = torch.bernoulli(predicts.data.squeeze())
                        indices = torch.nonzero(sampled_actions)

                        if len(indices) == 0:
                            continue
                        # print ('Selected %d/%d samples'%(len(indices), len(labels)))
                        count += len(indices)

                        selected_inputs = inputs[indices.squeeze()].view(len(indices), *inputs.size()[1:])
                        augmentations = {
                                2: TF.hflip,
                                3: TF.adjust_brightness,
                                4: TF.adjust_contrast,
                                }
                        to_PIL = torchvision.transforms.ToPILImage()
                        to_tensor = torchvision.transforms.ToTensor()
                        # select input and/or select data augmentation
                        if self.is_augment_teacher: # select data augmentations
                            assert (selected_inputs.shape[0] == len(indices))
                            for i, action in enumerate(sampled_actions[indices]):
                                temp_img = to_PIL(selected_inputs[i].cpu())
                                if action == 2:
                                    temp_img = TF.hflip(temp_img)
                                elif action == 3:
                                    temp_img = TF.adjust_brightness(temp_img, random.random()+0.5)
                                elif action == 4:
                                    temp_img = TF.adjust_contrast(temp_img, random.random()+0.5)
                                selected_inputs[i] = to_tensor(temp_img).cuda()
                            #inputs[indices.squeeze()] # the inputs to be augmented
                            #sampled_actions[indices] # the sampled augmentations
                            # (0=>no select,1=>select w/ no aug, 2=>, horizontal flip, 3=> adjust brightness, 4=> adjust contrast)
                        #else:
                        #    selected_inputs = inputs[indices.squeeze()].view(len(indices), *inputs.size()[1:])
                        selected_labels = labels[indices.squeeze()].view(-1, 1)
                        input_pool.append(selected_inputs)
                        label_pool.append(selected_labels)

                        if self.is_augment_teacher:
                            actions.append(action_log_probs)
                            batch_actions.append(action_log_probs[indices].view(-1))
                        else:
                            actions.append(torch.log(predicts.squeeze())*to_var(sampled_actions-0.5)*2)
                            batch_actions.append(torch.log(predicts.squeeze()[indices]))
                        if count >= M: # teachers have selected M samples; mini batch completed
                            effective_train_data += count
                            break
                    if count >= M:
                        break

                # ================== prepare training data =============
                inputs = torch.cat(input_pool, 0)
                labels = torch.cat(label_pool, 0)
                '''
                label_hist = torch.histc(labels.detach(), bins=num_classes, min=0, max=num_classes-1).cpu().numpy()
                if label_counts is None:
                    #label_counts = label_hist
                    label_counts = labels.detach().cpu().numpy()
                else:
                    #label_counts += label_hist
                    label_counts = np.append(label_counts, labels.detach().cpu().numpy())
                '''
                label_counts = np.append(label_counts, labels.detach().cpu().numpy())
                st_configs = {
                    'dataloader': to_generator([inputs, labels]),
                    'optimizer': student_optimizer,
                    'current_epoch': student_updates,
                    'total_epochs': 0,
                    'logger': logger,
                    'policy_step': teacher_updates
                }
                # ================= feed the selected batch ============
                train_loss, gradient_samples_train = student.fit(st_configs)
                training_loss_history.append(train_loss)
                student_updates += 1
                student_lr_scheduler(student_optimizer, student_updates)
                # ================ test on dev set =====================
                st_configs['dataloader'] = dev_dataloader
                acc, val_loss, gradient_samples_dev = student.val(st_configs)
                best_acc_on_dev = acc if best_acc_on_dev < acc else best_acc_on_dev
                logger.info('Stage [%d], Policy Steps: [%d] Test on Dev: Iteration [%d], accuracy: %5.4f, best: %5.4f, '
                            'loss: %5.4f' % (0, teacher_updates, student_updates, acc, best_acc_on_dev, val_loss))
                val_loss_history.append(val_loss)
                # ++++ gradient similarity reward ++++++

                # reformat train/dev gradients to be flat vector
                # TODO TODO TODO
                # fix this; getting stuck on 261
                train_grads = gradient_samples_train
                dev_grads = gradient_samples_dev
                '''
                train_grads = []
                for g in gradient_samples_train:
                    flat_grad = torch.Tensor().cuda()
                    for layer in g:
                        flat_grad = torch.cat((flat_grad, layer.view(-1)))
                    train_grads.append(flat_grad)
                dev_grads = []
                for g in gradient_samples_dev:
                    flat_grad = torch.Tensor().cuda()
                    for layer in g:
                        flat_grad = torch.cat((flat_grad, layer.view(-1)))
                    dev_grads.append(flat_grad)
                '''

                # flatten actions
                flat_actions = torch.cat(batch_actions)
                '''
                flat_actions = torch.Tensor().cuda()
                for a in batch_actions:
                    flat_actions = torch.cat((flat_actions,a))
                '''
                assert len(flat_actions) == len(train_grads)

                # calculate loss for teacher from dense reward
                cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                grad_loss = 0
                for jj in range(len(flat_actions)):
                    rand_dev_index = np.random.randint(len(dev_grads))
                    # get dense teacher reward from cosine similarity of training sample gradient and dev batch gradients
                    grad_reward = cos(train_grads[jj], dev_grads[rand_dev_index])
                    # policy gradient loss
                    grad_loss -= flat_actions[jj] * grad_reward.detach()

                if self.is_augment_teacher:
                    action_counts_list = []
                    for i in range(teacher.output_dim):
                        action_counts_list[i] = np.sum(action_counts==i)/len(action_counts)
                    grad_loss += np.linalg.norm(action_counts_list)

                teacher_optimizer.zero_grad()

                #s_grad = []
                #for p in student.parameters():
                #    s_grad.append(p.grad)
                grad_loss.backward(retain_graph=True)
                #s2_grad = []
                #for p in student.parameters():
                #    s2_grad.append(p.grad)

                teacher_optimizer.step()
                del train_grads
                del dev_grads
                del gradient_samples_dev
                del gradient_samples_train
                torch.cuda.empty_cache()
                ##########################################
                #########################################
                # ============== check if reach the expected accuracy or exceeds the max_t ==================
                #if i_tau % int(len(teacher_dataloader)*teacher_dataloader.batch_size/M) == 0: # one epoch for teacher
                if (i_tau+1) % 50 == 0:
                    writer.add_histogram('label_counts', label_counts, i_teacher, bins=10)
                    label_counts_dict = {}
                    for i in range(num_classes):
                        label_counts_dict['class_{}'.format(i)] = np.sum(label_counts==i)/len(label_counts)
                    writer.add_scalars('label_counts_scalars', label_counts_dict, i_teacher)
                    label_counts = np.array([])

                    if self.is_augment_teacher:
                        action_counts_dict = {}
                        for i in range(teacher.output_dim):
                            action_counts_dict['action_{}'.format(i)] = np.sum(action_counts==i)/len(action_counts)
                        writer.add_scalars('action_counts', action_counts_dict, i_teacher)
                        action_counts = np.array([])
                #    # validate teacher
                #    writer.add_text
                if acc >= tau or i_tau == max_t:
                    # TODO: add writer text about changing tau threshold
                    tau += (3/num_classes) * (1-tau)
                    num_steps_to_achieve.append(i_tau)
                    teacher_optimizer.zero_grad()

                    reward = -math.log(i_tau/max_t)
                    baseline = 0 if len(rewards) == 0 else 0.8*baseline + 0.2*reward
                    last_reward = 0 if len(rewards) == 0 else rewards[-1]

                    if last_reward >= reward:
                        non_increasing_steps += 1
                    else:
                        non_increasing_steps = 0

                    loss = -sum([torch.sum(_) for _ in actions])*(reward - baseline)
                    #print ('='*80)
                    #print (actions[0])
                    #print ('='*80)
                    logger.info('Policy: Iterations [%d], stops at %d/%d to achieve %5.4f, loss: %5.4f, '
                                'reward: %5.4f(%5.4f)'
                                %(teacher_updates, i_tau, max_t, acc, loss.cpu().item(), reward, baseline))
                    rewards.append(reward)
                    loss.backward()
                    teacher_optimizer.step()
                    #for name, param in teacher.named_parameters():
                    #    print (name, param)
                    teacher_updates += 1
                    teacher_lr_scheduler(teacher_optimizer, teacher_updates)

                    # ====== validated teacher =======
                    writer.add_text('teacher_update_{}'.format(teacher_updates), 'teacher finished at iteration: {}, effective training samples: {}'.format(i_tau, effective_train_data), i_tau)
                    val_curves = self.val_teacher(configs)

                    fig = plt.figure()
                    plt.plot(*zip(*val_curves))
                    plt.xlabel('effective training samples')
                    plt.ylabel('student test set accuracy')
                    writer.add_figure('val_curves_'.format(teacher_updates), fig, i_teacher)
                    for effective_num, acc in val_curves:
                        #writer.add_scalar('teacher_val_curves_{}'.format(teacher_updates), acc, effective_num)
                        writer.add_scalars('teacher_val_curves', {'teacher_step_{}'.format(str(teacher_updates)):acc}, effective_num)
                    # ========= reinitialize the student network =========
                    overloaded_init_params(self.student_net)
                    student_updates = 0
                    best_acc_on_dev = 0
                    print ('Initialized the student net\'s parameters')
                    # ========== break for next batch ====================
                    break


            # ==================== policy converged (stopping criteria) ==
            if non_increasing_steps >= max_non_increasing_steps:
                torch.save({'num_steps_to_achieve': num_steps_to_achieve}, './tmp/curve_stage_%d.pth.tar' % 0)
                print(num_steps_to_achieve)
                return num_steps_to_achieve
                # if pointer + 1 == len(tau_list):
                #     # logger.info()
                #     torch.save({'num_steps_to_achieve':num_steps_to_achieve}, './tmp/stage_%d.pth.tar'%(pointer))
                #     print (num_steps_to_achieve)
                #     return num_steps_to_achieve
                # else:
                #     logger.info('*******Going into the next stage[' + str(pointer + 1) + ']***********')
                #     torch.save({'num_steps_to_achieve': num_steps_to_achieve}, './tmp/stage_%d.pth.tar' % (pointer))
                #     print (num_steps_to_achieve[pointer])
                #     rewards = []
                #     training_loss_history = []
                #     val_loss_history = []
                #     non_increasing_steps = 0
                #     student_updates = 0
                #     teacher_updates = 0
                #     best_acc_on_dev = 0
                #     pointer += 1

    def val_teacher(self, configs):
        # TODO: test for the policy. Plotting the curve of #effective_samples-test_accuracy
        '''
        :param configs:
            Required:
                state_func
                dataloader: student/dev/test
                optimizer: student
                lr_scheduler: student
                logger
            Optional:
                threshold
                M
                num_classes
                max_t
                (Note: should be consistent with training)
        :return:
        '''
        teacher = self.teacher_net
        # ==================== train student from scratch ============
        init_params(self.student_net)
        student = self.student_net
        # ==================== fetch configs [optional] ===============
        threshold = configs.get('threshold', 0.5)
        M = configs.get('M', 128)
        num_classes = configs.get('num_classes', 10)
        max_t = configs.get('max_t', 50000)
        # =================== fetch configs [required] ================
        state_func = configs['state_func']
        student_dataloader = configs['dataloader']['student']
        dev_dataloader = configs['dataloader']['dev']
        test_dataloader = configs['dataloader']['test']
        student_optimizer = configs['optimizer']['student']
        student_lr_scheduler = configs['lr_scheduler']['student']
        logger = configs['logger']
        vae = configs['vae']

        # ================== init tracking history ====================
        training_loss_history = []
        val_loss_history = []

        student_updates = 0
        best_acc_on_dev = 0
        best_acc_on_test = 0
        i_tau = 0
        effective_num = 0
        effnum_acc_curves = []
        student_epoch = 0
        student_iter = 0

        action_counts = np.array([])
        while i_tau < max_t:
            i_tau += 1
            count = 0
            input_pool = []
            label_pool = []
            # ================== collect training batch ============
            for idx, (inputs, labels) in enumerate(student_dataloader):
                inputs = to_var(inputs)
                labels = to_var(labels)
                state_configs = {
                    'num_classes': num_classes,
                    'labels': labels,
                    'inputs': inputs,
                    'student': student,
                    'current_iter': i_tau,
                    'max_iter': max_t,
                    'train_loss_history': training_loss_history,
                    'val_loss_history': val_loss_history,
                    'use_vae': self.is_vae_teacher,
                    'vae': vae
                }
                states, vae_z = state_func(state_configs)  # TODO: implement the function for computing state
                _inputs = {'input': states, 'vae_z':vae_z}
                predicts = teacher(_inputs, None)

                if self.is_augment_teacher: # data augment teacher
                    teacher_actions = torch.argmax(predicts, dim=1) # n values (in the range of 0:action_space)
                    action_counts = np.append(action_counts, teacher_actions.detach().cpu().numpy())
                    # (0=>no select,1=>select w/ no aug, 2=>, horizontal flip, 3=> adjust brightness, 4=> adjust contrast)
                    # action_log_probs = m.log_prob(sampled_actions)
                    indices = torch.nonzero(teacher_actions)
                else:
                    indices = torch.nonzero(predicts.data.squeeze() >= threshold)

                if len(indices) == 0:
                    continue
                count += len(indices)
                # selected_inputs = torch.gather(inputs, 0, indices.squeeze()).view(len(indices),
                #                                                                  *inputs.size()[1:])
                # selected_labels = torch.gather(labels, 0, indices.squeeze()).view(-1, 1)
                # import pdb
                # pdb.set_trace()
                selected_inputs = inputs[indices.squeeze()].view(len(indices), *inputs.size()[1:])

                to_PIL = torchvision.transforms.ToPILImage()
                to_tensor = torchvision.transforms.ToTensor()
                # select input and/or select data augmentation
                if teacher.output_dim > 1: # select data augmentations
                    assert (selected_inputs.shape[0] == len(indices))
                    for i, action in enumerate(teacher_actions[indices]):
                        temp_img = to_PIL(selected_inputs[i].cpu())
                        if action == 2:
                            temp_img = TF.hflip(temp_img)
                        elif action == 3:
                            temp_img = TF.adjust_brightness(temp_img, random.random()*2)
                        elif action == 4:
                            temp_img = TF.adjust_contrast(temp_img, random.random()*2)
                        selected_inputs[i] = to_tensor(temp_img).cuda()
                    #inputs[indices.squeeze()] # the inputs to be augmented
                    #sampled_actions[indices] # the sampled augmentations
                    # (0=>no select,1=>select w/ no aug, 2=>, horizontal flip, 3=> adjust brightness, 4=> adjust contrast)
                #else:
                #    selected_inputs = inputs[indices.squeeze()].view(len(indices), *inputs.size()[1:])

                selected_labels = labels[indices.squeeze()].view(-1, 1)

                input_pool.append(selected_inputs)
                label_pool.append(selected_labels)
                if count >= M:
                    effective_num += count
                    break

            # ================== prepare training data =============
            if len(input_pool) == 0:
                print("Teacher selected 0 training inputs for student, probably an error!")
                break
            inputs = torch.cat(input_pool, 0)
            labels = torch.cat(label_pool, 0)
            st_configs = {
                'dataloader': to_generator([inputs, labels]),
                'optimizer': student_optimizer,
                'current_epoch': student_updates,
                'total_epochs': -1,
                'logger': logger
            }
            # ================= feed the selected batch ============
            train_loss = student.fit(st_configs, return_gradients=False)
            training_loss_history.append(train_loss)
            student_updates += 1
            student_lr_scheduler(student_optimizer, student_updates)

            student_iter += 1

            # TODO: add epoch based validation
            if i_tau % int(len(student_dataloader)*student_dataloader.batch_size/M) == 0: # one epoch for student
                logger.info('Teacher validation/ student validation: epoch {}'.format(student_epoch))
                student_epoch += 1
                student_iter = 0

                # ================ test on dev set =====================
                st_configs['dataloader'] = dev_dataloader
                acc, val_loss = student.val(st_configs, return_gradients=False)
                best_acc_on_dev = acc if best_acc_on_dev < acc else best_acc_on_dev
                #logger.info('Test on Dev: Iteration [%d], accuracy: %5.4f, best: %5.4f' % (student_updates,acc, best_acc_on_dev))
                logger.info('Test on Dev: accuracy: %5.4f, best: %5.4f' % (acc, best_acc_on_dev))
                val_loss_history.append(val_loss)

                # =============== test on test set ======================
                st_configs['dataloader'] = test_dataloader
                acc, test_loss = student.val(st_configs, return_gradients=False)
                best_acc_on_test = acc if best_acc_on_test < acc else best_acc_on_test
                #logger.info('Testing Set: Iteration [%d], accuracy: %5.4f, best: %5.4f' % (student_updates, acc, best_acc_on_test))
                logger.info('Testing on Test: accuracy: %5.4f, best: %5.4f' % (acc, best_acc_on_test))
                effnum_acc_curves.append((effective_num, acc))
        return effnum_acc_curves
