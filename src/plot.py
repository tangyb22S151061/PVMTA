import pandas as pd
import matplotlib.pyplot as plt

flag = input('是否修改了模型架构Y/N:')
# 提示用户输入文件名
for victim in ['fashionmnist','svhn','cifar10','gtsrb']: 
    for attack in ['maze','knockoff']:
        
        budget = '_30.0M'

        if victim == 'fashionmnist' and flag =='N':
            model_arch = 'lenet'
        else:
            if victim == 'fashionmnist':
                model_arch = 'res20_mnist'
            else:
                model_arch = 'res20'

        if attack == 'maze':
            x_column_name = 'queries'
            file_name = '/home/tyb/MAZE/logs'+ '/' + victim + '/' + model_arch + '/' + 'csv' + '/'+ attack + budget +'.csv'
        else :
            x_column_name = 'epochs'
            file_name = '/home/tyb/MAZE/logs'+ '/' + victim + '/' + model_arch + '/' + 'csv' + '/'+ attack + 'nets' +'.csv'
        y_column_name = 'accuracy'

        # 读取csv文件
        df = pd.read_csv(file_name)

        # 获取指定列的数据
        x_data = df[x_column_name]
        y_data = df[y_column_name]

        
        if attack == 'maze':
            plt.xlabel('queries(M)')
            plt.ylabel('acc(%)')
            plt.title( victim + '_' + model_arch + '_' + attack)
            plt.plot(x_data,y_data)
            image_path = '/home/tyb/MAZE/logs'+ '/' +  victim + '/' + model_arch + '/' + 'csv' + '/'+ attack + budget + '.png'
        else :
            plt.legend(loc = 'best')
            plt.xlabel('epochs')
            plt.ylabel('acc(%)')
            plt.title( victim + '_' + model_arch + '_' + attack + 'nets')
            plt.plot(x_data,y_data)
            image_path = '/home/tyb/MAZE/logs'+ '/' +  victim + '/' + model_arch + '/' + 'csv' + '/'+ attack + 'nets' + '.png'
        print(image_path)
        # 绘制图像
        plt.savefig(image_path)
        plt.clf()