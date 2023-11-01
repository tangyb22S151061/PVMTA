from tqdm import tqdm
import os
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from . import attack_utils
from utils.helpers import test, blackbox
import wandb
from datasets import get_dataset



def knockoff_para(args, T, S, test_loader, tar_acc, T_poison=None, delta = 0.6, savedir = "", K = 50, M = 20, op = 0):
    T.eval()
    T.id = 'user'
    S.train()

    if T_poison is not None:
        T_poison.eval()

    sur_data_loader, _ = get_dataset(args.dataset_sur, batch_size = args.batch_size)
    
    if args.opt == 'sgd':
        optS = optim.SGD(S.parameters(), lr=args.lr_clone, momentum=0.9, weight_decay=5e-4)
        schS = optim.lr_scheduler.CosineAnnealingLR(optS, args.epochs)
    else:
        optS = optim.Adam(S.parameters(), lr=args.lr_clone, weight_decay=5e-4)
        schS = optim.lr_scheduler.CosineAnnealingLR(optS, args.epochs)
    
    if op == 0:
        results = {'delta': [], 'accuracy': [], 'accuracy_x': []}
    elif op == 1:
        results = {'K': [], 'accuracy': [], 'accuracy_x': []}
    else:
        results = {'M': [], 'accuracy': [], 'accuracy_x': []}

    print('== Constructing Surrogate Dataset ==')
    sur_ds = []
    for data, _ in tqdm(sur_data_loader, ncols=100, leave=True):
        data = data.to(args.device)
        # 若由AM进行防御，则返回包含T_poison的结果
        if T_poison is not None:
            Tout = blackbox(data, T, T_poison)
        else:
            Tout = T(data)
        # 若被PRADA检测出，构建转移集行为被终止
        if Tout == 'Detected by PRADA':
            print('U\'ve been detected!')
            break
        Tout = F.softmax(Tout, dim=1)
        batch = [(a, b) for a, b in zip(data.cpu().detach().numpy(), Tout.cpu().detach().numpy())]
        sur_ds += batch
    sur_dataset_loader = torch.utils.data.DataLoader(sur_ds, batch_size=args.batch_size, num_workers=4, shuffle=True)

    print('\n== Training Clone Model ==')

    for epoch in range(1, args.epochs+1):
        S.train()
        train_loss, train_acc = attack_utils.train_soft_epoch(S, args.device, sur_dataset_loader, optS)
        test_loss, test_acc = test(S, args.device, test_loader)
        tar_acc_fraction = test_acc/tar_acc
        print('Epoch: {} Loss: {:.4f} Train Acc: {:.2f}% Test Acc: {:.2f} ({:.2f}x)%\n'.format(epoch, train_loss, train_acc, test_acc, tar_acc_fraction))
        wandb.log({'Train Acc': train_acc, 'Test Acc': test_acc, "Train Loss": train_loss})
        if schS:
            schS.step()
        
        if op == 0:
            results['delta'].append(delta)
        elif op == 1:
            results['K'].append(K)
        else:
            results['M'].append(M)
            
        results['accuracy'].append(test_acc)
        results['accuracy_x'].append(tar_acc_fraction)
    
    df = pd.DataFrame(data=results)
    filename = savedir + '/knockoffnets.csv'
    if os.path.isfile(filename):
    # 如果文件存在，不写入列名
        df.tail(1).to_csv(filename, mode='a', header=False, index=False)
    else:
    # 如果文件不存在，写入列名
        df.tail(1).to_csv(filename, mode='a', header=True, index=False)
    return