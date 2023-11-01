from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from . import attack_utils
from utils.helpers import test, blackbox
import wandb
from datasets import get_dataset



def knockoff_conf(args, T, S, test_loader, tar_acc, T_poison=None):
    T.eval()
    T.id = 'user'
    S.train()

    if args.attack == "knockoff" and args.dataset_sur == "cifar100":
        T.alpha = torch.tensor(0.85)

    if T_poison is not None:
        T_poison.eval()

    sur_data_loader, _ = get_dataset(args.dataset_sur, batch_size = args.batch_size)
    

    print('== Constructing Surrogate Dataset ==')
    sur_ds = []
    # 新增一个列表，用于保存查询样本的标号
    sur_idx = []
    # 新增一个列表，用于保存softmax的最大项
    sur_max = []
    for data, target in tqdm(sur_data_loader, ncols=100, leave=True):
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
        # 计算softmax的最大项，并添加到列表中
        max_val, _ = torch.max(Tout, dim=1)
        sur_max += max_val.cpu().detach().numpy().tolist()
        # 添加查询样本的标号到列表中
        sur_idx += target.cpu().detach().numpy().tolist()
        batch = [(a, b) for a, b in zip(data.cpu().detach().numpy(), Tout.cpu().detach().numpy())]
        sur_ds += batch
    sur_dataset_loader = torch.utils.data.DataLoader(sur_ds, batch_size=args.batch_size, num_workers=4, shuffle=True)

    savedir = '{}/{}/{}/'.format(args.logdir, args.dataset, args.model_tgt)
    

    # 新增一个文件夹，用于保存softmax的最大项和查询样本的标号
    savedir_sur = savedir + 'sur/'
    if not os.path.exists(savedir_sur):
        os.makedirs(savedir_sur)
    
    # 保存softmax的最大项和查询样本的标号到同一个.csv文件中
    # 创建一个数据框，用于存储softmax的最大项和查询样本的标号
    sur_df = pd.DataFrame({'sur_max': sur_max, 'sur_idx': sur_idx})
    # 保存数据框到.csv文件中
    sur_df.to_csv(savedir_sur + '/sur.csv', index=False)
    
    return
