import torch
import math
import sys
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
import numpy as np
eps=1e-7
# from ..utils.simutils import logs
from torch.autograd import Variable
from torch import autograd
import itertools
import torch.optim as optim
import random
#import kornia
import copy
import seaborn as sns
import time
tanh = nn.Tanh()
import pandas as pd
from cleverhans.future.torch.attacks import fast_gradient_method, projected_gradient_descent

def CXE_unif(logits):
    # preds = torch.log(logits) # convert to logits
    cxe = -(logits.mean(1) - torch.logsumexp(logits, dim=1)).mean()
    return cxe

# 待检测性质的黑盒API
def blackbox(x, model, model_posion):
    
    output = model(x)
    output_softmax = F.softmax(output, dim=1)
    y_max, _ = torch.max(output_softmax, dim=1)

    # 计算α值
    alpha = 1/(1+torch.exp(10000*(y_max.detach()-0.6)))
    alpha = alpha.unsqueeze(1)

    # 计算毒化模型的输出
    output_poison = model_posion(x)

    # 计算新的输出
    y_prime = (1-alpha) * output + alpha * output_poison

    return y_prime


def train_epoch(model, device, train_loader, opt, args, disable_pbar=False):
    model.train()
    correct = 0
    train_loss = 0
    # 损失进行平均而非求和
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, ncols=80, disable = disable_pbar, leave=False)):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = model(data)
        loss = criterion(output, target) 

        if args.adv_train:
            niter = 10
            data_adv = projected_gradient_descent(model, data, args.eps_adv, args.eps_adv/niter, niter, np.inf)
            output_adv = model(data) 
            loss += criterion(output_adv, target)  

        
        loss.backward()
        train_loss += loss
        opt.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader)
    train_acc = correct * 100. / len(train_loader.dataset)
    return train_loss, train_acc

def train_epoch_AM(model, device, train_loader, train_oe_loader, opt, args, model_poison=None, optimizer_poison=None, disable_pbar=False):
    model.train()
    correct = 0
    train_loss = 0
    #tmp
    oe_lamb = 0.1
    # 损失进行平均而非求和
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, ncols=80, disable = disable_pbar, leave=False)):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = model(data)
        loss = criterion(output, target) 

        # 处理分布外数据
        data_oe, _ = next(iter(train_oe_loader))
        data_oe = data_oe.to(device)
        output_oe = model(data_oe)
        loss_oe = CXE_unif(output_oe)
        loss += loss_oe * oe_lamb

        if args.adv_train:
            niter = 10
            data_adv = projected_gradient_descent(model, data, args.eps_adv, args.eps_adv/niter, niter, np.inf)
            output_adv = model(data) 
            loss += criterion(output_adv, target)  

        # 训练毒化模型
        if model_poison is not None and optimizer_poison is not None:
            
            inputs_all = torch.cat([data, data_oe])
            outputs_all = model(inputs_all)
            _, targets_all = torch.max(outputs_all.detach(), dim=1)
            outputs_poison = model_poison(inputs_all)
            outputs_poison_softmax = F.softmax(outputs_poison, dim =1)
            outputs_comp = torch.log(1-outputs_poison_softmax +1e-7)
            loss_poison = criterion(outputs_comp, targets_all)
            loss_poison.backward()
            optimizer_poison.step()

        
        loss.backward()
        train_loss += loss
        opt.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader)
    train_acc = correct * 100. / len(train_loader.dataset)
    return train_loss, train_acc

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct * 100. / len(test_loader.dataset)
    model.train()
    return test_loss, test_acc

def test_AM(model, device, test_loader, model_posion=None):
    model.eval()
    if model_posion is not None:
        model_posion.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = blackbox(x=data, model=model, model_posion=model_posion)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct * 100. / len(test_loader.dataset)

    model.train()
    if model_posion is not None:
        model_posion.train()

    return test_loss, test_acc
'''
def test_defender(model, device, test_loader):
    test_loss = 0
    correct = 0
    # 衰减温度
    # T = torch.tensor(1)
    # 温度下降因子
    T_decay = torch.tensor(0.9)
    # 置信度阈值
    alpha = 0.6
    # 恶性样本查询数
    Q = 0
    # 恶性用户判定阈值
    Q_thre = 1
    # 噪声大小比例
    noise_frac = torch.tensor(1e-3)
    # 增加的温度比例
    tep_frac = torch.tensor(1e-4)

    flag = True 
    model.T = torch.tensor(1.0)
    model.T = model.T.to(torch.float)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # 获得logits,logits_fake
            model.eval()
            logits_true, logits_fake = model(data)
            output = logits_true
            

            logits_fake = F.softmax(logits_fake)
            logits_max = torch.max(logits_fake)
            logits_max = logits_max.item()

            logits_true = F.softmax(logits_true)
            logits_max_true = torch.max(logits_true)
            logits_max_true = logits_max_true.item()

            print("当前温度:",model.T.item(),"\t","原本最大logits项:",logits_max_true,"\t","虚假最大logits项:",logits_max,"\n")
            
            # 恶性样本计数
            if logits_max_true <= alpha:
                Q += 1
            
            # 恶性用户判定
            if Q >= Q_thre:
                if logits_max_true <= alpha:
                    model.T *= T_decay
                    noise_mean = model.T * noise_frac
                    noise_std = noise_mean * noise_frac
                    noise = noise_std * torch.randn([]) + noise_mean
                    model.T += noise
                    print(noise, model.T)
                    model.eval()
                else:
                    model.T += model.T * tep_frac
            
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    test_acc = correct * 100. / len(test_loader.dataset)
    model.train()
    return test_loss, test_acc
'''