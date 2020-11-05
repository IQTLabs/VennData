from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from misc.utils import to_var
import numpy as np
import time

from core.models.resnet import ResNet34

class StudentNetwork(nn.Module):
    def __init__(self, configs, device):
        '''
        Descriptions:
            Some details of the paper:
            1. Use ResNet as the CNN student model, or use LSTM as the RNN student model.
            2. SGD for CNN, Adam for RNN.
            3.

        :param configs:
        '''
        super(StudentNetwork, self).__init__()
        self.base_model = configs['base_model']
        self.evaluator = configs['evaluator']
        self.device = device

    def forward(self, data, configs):
        inputs, labels = data['inputs'], data['labels']
        # import pdb
        # pdb.set_trace()
        predicts = self.base_model(inputs)
        eval_res = self.evaluator(predicts, labels)
        return predicts, eval_res

    def fit(self, configs, return_gradients=True):
        self.base_model.train()
        dataloader, optimizer = configs['dataloader'], configs['optimizer']
        try:
            flag = True
            total_steps = len(dataloader)
        except:
            flag = False
            total_steps = 1
        current_epoch = configs['current_epoch']
        total_epochs = configs['total_epochs']
        teacher_updates = configs.get('policy_step', -1)
        logger = configs['logger']

        all_correct = 0
        all_samples = 0
        loss_average = 0
        gradient_samples = []

        for idx, (inputs, labels) in enumerate(dataloader):
            #print('input shape - ',inputs.shape)
            optimizer.zero_grad()
            if flag:
                inputs = to_var(inputs).to(self.device)
                labels = to_var(labels).to(self.device)
            predicts = self.base_model(inputs)

            eval_res = self.evaluator(predicts, labels)
            num_correct = eval_res['num_correct']
            num_samples = eval_res['num_samples']
            # logger.info('num_samples %d, num_correct %d'%(num_samples, num_correct))
            loss = eval_res['loss']
            all_correct += num_correct
            all_samples += num_samples
            full_loss = eval_res['full_loss']
            #print(full_loss.shape)
            #print('loss = ',loss)
            #print('full loss/n = ',torch.sum(full_loss)/len(full_loss))
            #print('full loss shape = ',full_loss.shape)
            if return_gradients:
                for l in full_loss:
                    #print('len gradient samples = ', len(gradient_samples))
                    gradients = torch.autograd.grad(l, (self.base_model.parameters()), create_graph=False, retain_graph=True)
                    gradient_samples.append(torch.cat([g.detach().view(-1) for g in gradients]))
            loss.backward()
            optimizer.step()
            #print('batch accuracy = ',num_correct/num_samples)
            #logger.info('Policy Steps: [%d] Train: ----- Iteration [%d], loss: %5.4f, accuracy: %5.4f(%5.4f)' % (
            #    teacher_updates, current_epoch+1, loss.cpu().item(), num_correct/num_samples, all_correct/all_samples))
            logger.info('Policy: [%d] Train: -- Iter [%d], loss: %5.4f, minibatch-accuracy: %5.4f (%d batch size)' % (
                teacher_updates, current_epoch+1, loss.cpu().item(), num_correct/num_samples, num_samples))
            loss_average += loss.cpu().item()
        if return_gradients:
            return loss_average/total_steps, gradient_samples
        else:
            return loss_average/total_steps

    def val(self, configs, return_gradients=True):
        self.base_model.eval()
        dataloader = configs['dataloader']
        total_steps = len(dataloader)

        all_correct = 0
        all_samples = 0
        loss_average = 0
        gradient_samples = []
        for x in self.base_model.parameters():
            if x.grad is not None:
                x.grad.data.zero_()
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = to_var(inputs).to(self.device)
            labels = to_var(labels).to(self.device)
            predicts = self.base_model(inputs)
            eval_res = self.evaluator(predicts, labels)
            num_correct = eval_res['num_correct']
            num_samples = eval_res['num_samples']
            all_correct += num_correct
            all_samples += num_samples
            # logger.info('Eval: Epoch [%d/%d], Iteration [%d/%d], accuracy: %5.4f(%5.4f)' % (
            #    current_epoch, total_epochs, idx, total_steps, num_correct/num_samples, all_correct/all_samples))
            loss_average += eval_res['loss'].cpu().item()

            ### gradient similarity
            loss = eval_res['loss']


            #print('dev inputs shape = ',inputs.shape)
            if return_gradients:
                gradients = torch.autograd.grad(loss, (self.base_model.parameters()), create_graph=False, retain_graph=False)
                gradient_samples.append(torch.cat([g.detach().view(-1) for g in gradients]))
            #for x in self.base_model.parameters():
            #    if x.grad is not None:
            #        x.grad.data.zero_()
            #loss.backward()
            #loss.backward(create_graph=True, retain_graph=True)
            #print('one loop \n\n\n')
            #for par, g in zip(self.base_model.parameters(), gw):
            #    assert par.grad.shape == g.shape
            #    diff = par.grad - g
                #print('diff = ',diff[0])
            #    print('sum(diff) = ',torch.sum(torch.abs(diff.detach())))
                #print('par.grad = ',par.grad)
                #print('g = ',g)
        # print ('Total: %d, correct: %d', all_samples, all_correct)

        if return_gradients:
            return all_correct/all_samples, loss_average/total_steps, gradient_samples
        else:
            return all_correct/all_samples, loss_average/total_steps




