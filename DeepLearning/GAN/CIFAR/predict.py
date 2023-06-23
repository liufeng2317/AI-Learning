'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2023-02-11 16:40:26
* LastEditors: LiuFeng
* LastEditTime: 2023-02-11 22:52:13
* FilePath: /gan-cifar/predict.py
* Description: 
* Copyright (c) 2023 by ${git_name} email: ${git_email}, All Rights Reserved.
'''
import torch
import numpy as np
from model.generator import Generator, Generator_Transpose
from model.discriminator import DiscriminatorResnet, DiscriminatorLinear, DiscriminatorConv
from PIL import Image

if __name__ == '__main__':
    input_size = [3, 32, 32]
    in_channel = 64
    gen_para_path = './log/2023-02-11-17_52_12/epoch999-genLoss1.21-disLoss0.40-gen.pth'
    dis_para_path = './log/2023-02-11-17_52_12/epoch999-genLoss1.21-disLoss0.40-dis.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = Generator_Transpose(in_channel=in_channel).to(device)
    dis = DiscriminatorLinear(input_size=input_size).to(device)
    gen.load_state_dict(torch.load(gen_para_path, map_location=device))
    gen.eval()
    # 随机生成一组数据
    G_img = torch.randn([1, in_channel, 1, 1], requires_grad=False).to(device)
    # 放入网路
    G_pred = gen(G_img)
    G_dis = dis(G_pred)
    print('generator-dis:', G_dis)
    # 图像显示
    G_pred = G_pred[0, ...]
    G_pred = G_pred.detach().cpu().numpy()
    G_pred = np.array(G_pred * 255)
    G_pred = np.transpose(G_pred, [1, 2, 0])
    G_pred = Image.fromarray(np.uint8(G_pred))
    G_pred.show()
