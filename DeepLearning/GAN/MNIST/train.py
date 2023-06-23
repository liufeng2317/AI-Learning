'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2023-02-21 20:45:02
* LastEditors: LiuFeng
* LastEditTime: 2023-02-22 13:28:05
* FilePath: /GAN_LF/demo/MNIST/train.py
* Description: 
* Copyright (c) 2023 by ${git_name} email: ${git_email}, All Rights Reserved.
'''
import os
import time
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.utils import save_image
import sys 
from pathlib import Path
sys.path.append(os.path.abspath(r"../.."))
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mlxtend.data import loadlocal_mnist


if __name__=="__main__":
    input_size = [1, 28, 28]
    batch_size = 128
    Epoch = 1000
    GenEpoch = 1
    in_channel = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ##########################################################################
    class MNIST_Dataset(Dataset):
        def __init__(self,train_data_path,train_label_path,transform=None):
            train_data,train_label = loadlocal_mnist(
                images_path = train_data_path,
                labels_path = train_label_path
                )
            self.train_data = train_data
            self.train_label = train_label.reshape(-1)
            self.transform=transform
            
        def __len__(self):
            return self.train_label.shape[0] 
        
        def __getitem__(self,index):
            if torch.is_tensor(index):
                index = index.tolist()
            images = self.train_data[index,:].reshape((28,28))
            labels = self.train_label[index]
            if self.transform:
                images = self.transform(images)
            return images,labels

    train_data_path = "../../../../data/MNIST/train-images.idx3-ubyte"
    train_label_path = "../../../../data/MNIST/train-labels.idx1-ubyte"
    transform_dataset =transforms.Compose([
        transforms.ToTensor()]
    )
    MNIST_dataset = MNIST_Dataset(train_data_path=train_data_path,
                                train_label_path=train_label_path,
                                transform=transform_dataset)  
    MNIST_dataloader = DataLoader(dataset=MNIST_dataset,
                                batch_size=batch_size,
                                shuffle=True,drop_last=False)
    
    ########################################################################
    time_now = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))
    log_path = f'./log/{time_now}'
    os.makedirs(log_path)
    os.makedirs(f'{log_path}/image')
    os.makedirs(f'{log_path}/image/image_all')
    #######################################################################
    class Discriminator(nn.Module):
        def __init__(self,input_size,inplace=True):
            super(Discriminator,self).__init__()
            c,h,w = input_size
            self.dis = nn.Sequential(
                nn.Linear(c*h*w,512),  # 输入特征数为784，输出为512
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),  # 进行非线性映射
                
                nn.Linear(512, 256),  # 进行一个线性映射
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
                
                nn.Linear(256, 1),
                nn.Sigmoid()  # 也是一个激活函数，二分类问题中，
                # sigmoid可以班实数映射到【0,1】，作为概率值，
                # 多分类用softmax函数
            )
            
            
        def forward(self,x):
            b,c,h,w = x.size()
            x = x.view(b,-1)
            x = self.dis(x)
            x = x.view(-1)
            return x 

    class Generator(nn.Module):
        def __init__(self,in_channel):
            super(Generator,self).__init__() # 调用父类的构造方法
            self.gen = nn.Sequential(
                nn.Linear(in_channel, 128),
                nn.LeakyReLU(0.1),
                
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),
                
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.1),
                
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.1),
                
                nn.Linear(1024, 784),
                nn.Tanh()
            )

        def forward(self,x):
            res = self.gen(x)
            return res.view(x.size()[0],1,28,28)

    D = Discriminator(input_size=input_size)
    G = Generator(in_channel=in_channel)
    D.to(device)
    G.to(device)
    ####################################################################
    criterion = nn.BCELoss()
    D_optimizer = torch.optim.Adam(D.parameters(),lr=0.0003)
    G_optimizer = torch.optim.Adam(G.parameters(),lr=0.0003)
    D.train()
    G.train()
    gen_loss_list = []
    dis_loss_list = []

    for epoch in range(Epoch):
        with tqdm(total=MNIST_dataloader.__len__(),desc=f'Epoch {epoch+1}/{Epoch}')as pbar:
            gen_loss_avg = []
            dis_loss_avg = []
            index = 0
            for batch_idx,(img,_) in enumerate(MNIST_dataloader):
                img = img.to(device)
                # the output label
                valid = torch.ones(img.size()[0]).to(device)
                fake = torch.zeros(img.size()[0]).to(device)
                # Generator input
                G_img = torch.randn([img.size()[0],in_channel],requires_grad=True).to(device)
                # ------------------Update Discriminator------------------
                # forward
                G_pred_gen = G(G_img)
                G_pred_dis = D(G_pred_gen.detach())
                R_pred_dis = D(img)
                # the misfit
                G_loss = criterion(G_pred_dis,fake)
                R_loss = criterion(R_pred_dis,valid)
                dis_loss = (G_loss+R_loss)/2
                dis_loss_avg.append(dis_loss.item())
                # backward
                D_optimizer.zero_grad()
                dis_loss.backward()
                D_optimizer.step()
                # ------------------Update Optimizer------------------
                # forward
                G_pred_gen = G(G_img)
                G_pred_dis = D(G_pred_gen)
                # the misfit
                gen_loss = criterion(G_pred_dis,valid)
                gen_loss_avg.append(gen_loss.item())
                # backward
                G_optimizer.zero_grad()
                gen_loss.backward()
                G_optimizer.step()
                # save figure
                if index % 200 == 0 or index + 1 == MNIST_dataset.__len__():
                    save_image(G_pred_gen, f'{log_path}/image/image_all/epoch-{epoch}-index-{index}.png')
                index += 1
                # ------------------进度条更新------------------
                pbar.set_postfix(**{
                    'gen-loss': sum(gen_loss_avg) / len(gen_loss_avg),
                    'dis-loss': sum(dis_loss_avg) / len(dis_loss_avg)
                })
                pbar.update(1)
            save_image(G_pred_gen, f'{log_path}/image/epoch-{epoch}.png')
        filename = 'epoch%d-genLoss%.2f-disLoss%.2f' % (epoch, sum(gen_loss_avg) / len(gen_loss_avg), sum(dis_loss_avg) / len(dis_loss_avg))
        torch.save(G.state_dict(), f'{log_path}/{filename}-gen.pth')
        torch.save(D.state_dict(), f'{log_path}/{filename}-dis.pth')
        # 记录损失
        gen_loss_list.append(sum(gen_loss_avg) / len(gen_loss_avg))
        dis_loss_list.append(sum(dis_loss_avg) / len(dis_loss_avg))
        # 绘制损失图像并保存
        plt.figure(0)
        plt.plot(range(epoch + 1), gen_loss_list, 'r--', label='gen loss')
        plt.plot(range(epoch + 1), dis_loss_list, 'r--', label='dis loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(f'{log_path}/loss.png', dpi=300)
        plt.close(0)