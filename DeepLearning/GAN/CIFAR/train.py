import os
import time
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from model.generator import Generator, Generator_Transpose
from model.discriminator import DiscriminatorResnet, DiscriminatorLinear, DiscriminatorConv
from utils.utils import weights_init, weight_init
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 参数设置
    input_size = [3, 32, 32]
    batch_size = 128
    Epoch = 1000
    GenEpoch = 1
    in_channel = 64
    # 生成日志文件夹
    time_now = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))
    log_path = f'./log/{time_now}'
    os.makedirs(log_path)
    os.makedirs(f'{log_path}/image')
    os.makedirs(f'{log_path}/image/image_all')
    # 判断使用设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device: {device}')
    # 加载数据集
    cifar_train = datasets.CIFAR10('../../../../data/cifar', True,
                                    transform=transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.ToTensor()]
                                    ),
                                    download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)

    cifar_test = datasets.CIFAR10('../../../../data/cifar', False,
                                    transform=transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.ToTensor()]
                                    ),
                                    download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)
    # 定义模型，生成cifar-10建议使用Transpose作为生成网络，使用卷积层作为辨别网络
    # 要注意生成网络与辨别网络得匹配程度
    # 如果两个网络的损失都向0.693靠拢，则说明其中一个网络运行模式崩溃
    # 较好的gan网络能使得生成网络损失维持在1.0+左右，辨别网络损失维持较低的水平，即两个网络相互促进的过程
    gen = Generator_Transpose(in_channel=in_channel)
    dis = DiscriminatorConv(input_size=input_size)
    gen.apply(weight_init)
    dis.apply(weight_init)
    gen.to(device)
    dis.to(device)
    # 设置损失函数
    Loss = nn.CrossEntropyLoss()
    # 设置优化器
    opt_gen = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_dis = optim.Adam(dis.parameters(), lr=2e-4, betas=(0.5, 0.999))
    # 模型训练
    gen.train()
    dis.train()
    gen_loss_list = []       # 生成网路损失
    dis_loss_list = []       # 判别网络损失
    for epoch in range(Epoch):
        with tqdm(total=cifar_train.__len__(), desc=f'Epoch {epoch + 1}/{Epoch}') as pbar:
            gen_loss_avg = []
            dis_loss_avg = []
            index = 0       # 记录训练了多少个batch
            for batchidx, (img, _) in enumerate(cifar_train):
                # 获取数据
                img = img.to(device)
                # 获取标注向量
                valid = torch.ones(img.size()[0], dtype=torch.int64).to(device)
                fake = torch.zeros(img.size()[0], dtype=torch.int64).to(device)
                # 随机生成一组数据
                G_img = torch.randn([img.size()[0], in_channel, 1, 1], requires_grad=True).to(device)
                # ------------------更新判别器------------------
                # 前向计算
                G_pred_gen = gen(G_img)
                G_pred_dis = dis(G_pred_gen.detach())
                R_pred_dis = dis(img)
                # 计算损失
                G_loss = Loss(G_pred_dis, fake)
                R_loss = Loss(R_pred_dis, valid)
                dis_loss = (R_loss + G_loss) / 2
                dis_loss_avg.append(dis_loss.item())
                # 反向传播
                opt_dis.zero_grad()
                dis_loss.backward()
                opt_dis.step()
                # ------------------更新生成器------------------
                # 前向计算
                G_pred_gen = gen(G_img)
                G_pred_dis = dis(G_pred_gen)
                # 计算损失
                gen_loss = Loss(G_pred_dis, valid)
                gen_loss_avg.append(gen_loss.item())
                # 反向传播
                opt_gen.zero_grad()
                gen_loss.backward()
                opt_gen.step()
                # 保存过程图片
                if index % 100 == 0 or index + 1 == cifar_train.__len__():
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
        torch.save(gen.state_dict(), f'{log_path}/{filename}-gen.pth')
        torch.save(dis.state_dict(), f'{log_path}/{filename}-dis.pth')
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



