from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import math
import torch.nn as nn
import numpy as np
from collections import defaultdict


from core.student_network import StudentNetwork
from core.teacher_network import TeacherNetwork

from misc.utils import init_params, to_var

# ================== helper function ===============
def to_generator(data):
    yield data

# ==================================================


class MultiTeacherStudentModel(nn.Module):

    def __init__(self, configs):
        super(MultiTeacherStudentModel, self).__init__()
        self.configs = configs
        self.student_net = StudentNetwork(configs['student_configs'])
        self.teacher_nets = []
        for i in range(self.configs['num_classes']):
            self.teacher_nets.append(TeacherNetwork(configs['teacher_configs']).cuda())
        init_params(self.student_net)


    def fit_teacher(self, configs):
        '''
        :param configs:
            Required:
                state_func: [function] used to compute the state vector

                dataloader: [dict]
                    teacher: teacher training data loader
                    student: student training data loader
                    dev: [list] for testing the student model so as to compute reward for the teacher
                    test: student testing data loader

                optimizer: [dict]
                    teacher: [list] the optimizers for teachers
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
        teachers = self.teacher_nets
        student = self.student_net
        #print('parameter devices (teacher0, teacher1, student)')
        #print(next(teachers[0].parameters()).device)
        #print(next(teachers[1].parameters()).device)
        #print(next(student.parameters()).device)
        # ==================== fetch configs [optional] ===============
        max_t = configs['max_t']
        tau = configs['tau']
        M = configs['M']
        max_non_increasing_steps = configs['max_non_increasing_steps']
        num_classes = configs['num_classes']

        # =================== fetch configs [required] ================
        state_func = configs['state_func']
        teacher_dataloader = configs['dataloader']['teacher']
        dev_dataloaders = configs['dataloader']['dev']
        teacher_optimizers = configs['optimizer']['teacher']
        student_optimizer = configs['optimizer']['student']
        teacher_lr_scheduler = configs['lr_scheduler']['teacher']
        student_lr_scheduler = configs['lr_scheduler']['student']
        logger = configs['logger']

        # ================== init tracking history ====================
        rewards = []
        training_loss_history = []
        val_loss_history = []
        num_steps_to_achieve = []

        non_increasing_steps = 0
        student_updates = 0
        teacher_updates = 0
        best_acc_on_dev = 0
        while True:
            i_tau = 0
            actions = []

            def overloaded_init_params(x):
                init_params(x)
                # if pointer == 0:
                #    init_params(x)
                # else:
                #     file_name = './model/resnet34-%5.4f.pth.tar' % (tau_list[pointer - 1])
                #     logger.info('Loaded model from' + file_name)
                #     x.load_state_dict(torch.load(file_name)['state_dict'])

            while i_tau < max_t:
                i_tau += 1
                count = 0
                input_pool = []
                label_pool = []
                batch_actions = []
                batch_agents = []
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
                            'val_loss_history': val_loss_history
                        }
                        states = state_func(state_configs)  # TODO: implement the function for computing state
                        _inputs = {'input': states.detach()}
                        #predicts = teacher(_inputs, None)
                        predicts = teachers[idx%configs['num_classes']](_inputs, None)
                        sampled_actions = torch.bernoulli(predicts.data.squeeze())
                        indices = torch.nonzero(sampled_actions)

                        if len(indices) == 0:
                            continue
                        # print ('Selected %d/%d samples'%(len(indices), len(labels)))
                        count += len(indices)
                        selected_inputs = inputs[indices.squeeze()].view(len(indices), *inputs.size()[1:])
                        selected_labels = labels[indices.squeeze()].view(-1, 1)
                        input_pool.append(selected_inputs)
                        label_pool.append(selected_labels)
                        actions.append(torch.log(predicts.squeeze())*to_var(sampled_actions-0.5)*2)
                        for indi in indices:
                            batch_agents.append(indi)
                        batch_actions.append(torch.log(predicts.squeeze()[indices]))
                        if count >= M:
                            break
                    if count >= M:
                        break

                # ================== prepare training data =============
                inputs = torch.cat(input_pool, 0)
                labels = torch.cat(label_pool, 0)
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
                accs = []
                val_losses = []
                agent_gradients_devs = []
                #print(dev_dataloaders)
                for dev_data in dev_dataloaders:
                    st_configs['dataloader'] = dev_dataloaders[dev_data]
                    #print('dev_data = ',dev_data)
                    acc, val_loss, gradient_samples_dev = student.val(st_configs)
                    accs.append(acc)
                    val_losses.append(val_loss)
                    agent_gradients_devs.append(gradient_samples_dev)

                acc = np.mean(accs)
                val_loss = np.mean(val_losses)
                best_acc_on_dev = acc if best_acc_on_dev < acc else best_acc_on_dev
                logger.info('Stage [%d], Policy Steps: [%d] Test on Dev: Iteration [%d], accuracy: %5.4f, best: %5.4f, '
                            'loss: %5.4f' % (0, teacher_updates, student_updates, acc, best_acc_on_dev, val_loss))
                val_loss_history.append(val_loss)
                # ++++ gradient similarity reward ++++++

                # reformat train/dev gradients to be flat vector
                train_grads = []
                for g in gradient_samples_train:
                    flat_grad = torch.Tensor().cuda()
                    for layer in g:
                        flat_grad = torch.cat((flat_grad, layer.view(-1)))
                    train_grads.append(flat_grad)
                agent_dev_grads = []
                for i in range(num_classes):
                    dev_grads = []
                    for g in gradient_samples_dev:
                        flat_grad = torch.Tensor().cuda()
                        for layer in g:
                            flat_grad = torch.cat((flat_grad, layer.view(-1)))
                        dev_grads.append(flat_grad)
                    agent_dev_grads.append(dev_grads)

                # flatten actions
                flat_actions = torch.Tensor().cuda()
                for a in batch_actions:
                    flat_actions = torch.cat((flat_actions,a))
                assert len(flat_actions) == len(train_grads)

                # calculate loss for teacher from dense reward
                cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                grad_loss = 0
                for jj in range(len(flat_actions)):
                    rand_dev_index = np.random.randint(len(agent_dev_grads[batch_agents[jj]]))
                    # get dense teacher reward from cosine similarity of training sample gradient and dev batch gradients
                    grad_reward = cos(train_grads[jj], agent_dev_grads[batch_agents[jj]][rand_dev_index])
                    # policy gradient loss
                    grad_loss -= flat_actions[jj] * grad_reward.detach()

                for teacher_optimizer in teacher_optimizers:
                    teacher_optimizer.zero_grad()

                #s_grad = []
                #for p in student.parameters():
                #    s_grad.append(p.grad)
                grad_loss.backward(retain_graph=True)
                #s2_grad = []
                #for p in student.parameters():
                #    s2_grad.append(p.grad)

                for teacher_optimizer in teacher_optimizers:
                    teacher_optimizer.step()
                del train_grads
                del dev_grads
                del gradient_samples_dev
                del gradient_samples_train
                torch.cuda.empty_cache()
                ##########################################
                #########################################
                # ============== check if reach the expected accuracy or exceeds the max_t ==================
                if acc >= tau or i_tau == max_t:
                    tau += (3/num_classes) * (1-tau)
                    num_steps_to_achieve.append(i_tau)
                    for teacher_optimizer in teacher_optimizers:
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
                    #for name, param in teacher.named_parameters():
                    #    print (name, param)
                    teacher_updates += 1
                    for teacher_optimizer in teacher_optimizers:
                        teacher_optimizer.step()

                        teacher_lr_scheduler(teacher_optimizer, teacher_updates)

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

        # ================== init tracking history ====================
        training_loss_history = []
        val_loss_history = []

        student_updates = 0
        best_acc_on_dev = 0
        best_acc_on_test = 0
        i_tau = 0
        effective_num = 0
        effnum_acc_curves = []

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
                    'val_loss_history': val_loss_history
                }
                states = state_func(state_configs)  # TODO: implement the function for computing state
                _inputs = {'input': states}
                predicts = teacher(_inputs, None)

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
                selected_labels = labels[indices.squeeze()].view(-1, 1)
                input_pool.append(selected_inputs)
                label_pool.append(selected_labels)
                if count >= M:
                    effective_num += count
                    break

            # ================== prepare training data =============
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
            train_loss = student.fit(st_configs)
            training_loss_history.append(train_loss)
            student_updates += 1
            student_lr_scheduler(student_optimizer, student_updates)

            # ================ test on dev set =====================
            st_configs['dataloader'] = dev_dataloader
            acc, val_loss = student.val(st_configs)
            best_acc_on_dev = acc if best_acc_on_dev < acc else best_acc_on_dev
            logger.info('Test on Dev: Iteration [%d], accuracy: %5.4f, best: %5.4f' % (student_updates,
                                                                                       acc, best_acc_on_dev))
            val_loss_history.append(val_loss)

            # =============== test on test set ======================
            st_configs['dataloader'] = test_dataloader
            acc, test_loss = student.val(st_configs)
            best_acc_on_test = acc if best_acc_on_test < acc else best_acc_on_test
            logger.info('Testing Set: Iteration [%d], accuracy: %5.4f, best: %5.4f' % (student_updates,
                                                                                       acc, best_acc_on_test))
            effnum_acc_curves.append((effective_num, acc))
        return effnum_acc_curves
