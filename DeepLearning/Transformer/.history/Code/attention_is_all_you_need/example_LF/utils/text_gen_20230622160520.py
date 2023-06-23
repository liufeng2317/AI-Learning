'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2023-06-22 15:47:25
* LastEditors: LiuFeng
* LastEditTime: 2023-06-22 16:05:19
* FilePath: /Transformer/Code/attention_is_all_you_need/example_LF/utils/text_gen.py
* Description: 
* Copyright (c) 2023 by ${git_name} email: ${git_email}, All Rights Reserved.
'''
import torch 
from tqdm.auto import tqdm 

# From https://github.com/karpathy/nanoGPT/blob/master/train.py
def get_batch(data,sequence_length,batch_size,device='cpu'):
    device_type = device 
    ix = torch.randint(len(data)-sequence_length,(batch_size,))
    x = torch.stack([(data[i:i+sequence_length]) for i in ix])
    y = torch.stack([(data[i+1:i+1+sequence_length]) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to remove them to GPU asynchronously(non_blocking=True)
        x = x.pin_memory().to(device,non_blocking=True)
        y = y.pin_memory().to(device,non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

# training function
def train(
    model,
    dataset_train,
    optimizer,
    criterion,
    sequence_length,
    vocab_size,
    batch_size,
    device
):
    model.train()
    print("#"*50)
    print("Training")
    print("#"*50)
    
    counter = 0
    train_running_loss = 0.0
    
    for i in tqdm(range(0,dataset_train.size(0),sequence_length),total=int(dataset_train.size(0)/sequence_length)):
        counter += 1
        inputs,labels = get_batch(
            dataset_train,sequence_length,batch_size,device
        )
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(inputs)
        
        labels = labels.contiguous().view(-1)
        outputs = outputs.view(-1,vocab_size)
        
        # loss
        loss = criterion(
            outputs,
            labels.type(torch.int64)
        )
        train_running_loss += loss.item()
        
        # Backpropagation
        loss.backward()
        
        # update the optimizer parameters
        optimizer.step()
    
    # Loss and accuracy for the complete epoch
    epoch_loss = train_running_loss/counter
    return epoch_loss
