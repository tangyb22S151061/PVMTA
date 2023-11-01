import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
import random
import numpy as np
seed = 2020
random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.multiprocessing.set_sharing_strategy('file_system')

from utils.simutils.timer import timer
from utils.config import parser
from models.models import get_model
from datasets import get_dataset
# from utils.helpers import test, train_epoch
from utils.helpers import train_epoch_AM, test_AM

args = parser.parse_args()

wandb.init(project=args.wandb_project)
run_name = 'defender_{}_{}'.format(args.dataset, args.model_tgt)
wandb.run.name = run_name
wandb.run.save()

if args.device == 'gpu':
    import torch.backends.cudnn as cudnn
    cudnn.enabled = True
    cudnn.benchmark = True
    args.device = 'cuda'
else:
    args.device = 'cpu'


def train_defender():
    # f
    model = get_model(args.model_tgt, args.dataset, args.pretrained)
    # f'
    model_poison = get_model(args.model_tgt, args.dataset, args.pretrained)

    lr_gamma = 0.1
    lr_step = 50

    # 将模型放入指定运行设备上
    model = model.to(args.device)

    model_poison = model_poison.to(args.device)

    # 获得oe数据集和正常数据集
    train_loader, test_loader = get_dataset(args.dataset, args.batch_size, augment=True)
    train_oe_loader, test_oe_loader = get_dataset(args.dataset_oe, args.batch_size, augment=True)

    savedir = '{}/{}/{}/'.format(args.logdir, args.dataset, args.model_tgt)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    # 指定保存路径
    savepath1 = savedir + 'f.pt'
    savepath2 = savedir + 'f_poison.pt'

    sch = None

    # 设置原模型和毒化模型优化器和学习率调度器
    if args.opt == 'sgd': 
        opt = optim.SGD(model.parameters(), lr=args.lr_tgt, momentum=0.9, weight_decay=5e-4)
        sch = optim.lr_scheduler.StepLR(
            opt, step_size=lr_step, gamma=lr_gamma
        )
        if model_poison is not None:
            opt_poison = optim.SGD(model_poison.parameters(), lr=args.lr_tgt, momentum=0.9, weight_decay=5e-4)
            sch_poison = optim.lr_scheduler.StepLR(
            opt_poison, step_size=lr_step, gamma=lr_gamma
        )
    elif args.opt == 'adam': # and Adam for the rest
        opt = optim.Adam(model.parameters(), lr=args.lr_tgt)
    else:
        sys.exit('Invalid optimizer {}'.format(args.opt))

    
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch_AM(model, args.device, train_loader, train_oe_loader, opt, args, model_poison=model_poison, optimizer_poison=opt_poison, disable_pbar=False)
        test_loss, test_acc = test_AM(model, args.device, test_loader, model_poison)
        print('Epoch: {} Loss: {:.4f} Train Acc: {:.2f}% Test Acc: {:.2f}%\n'.format(epoch+1, train_loss, train_acc, test_acc))
        wandb.log({'Train Acc': train_acc, 'Test Acc': test_acc, "Train Loss": train_loss})
        if sch:
            sch.step()
        if sch_poison:
            sch_poison.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), savepath1)
            torch.save(model_poison.state_dict(),savepath2)

def main():
    timer(train_defender)
    exit(0)

if __name__ == '__main__':
    main()

