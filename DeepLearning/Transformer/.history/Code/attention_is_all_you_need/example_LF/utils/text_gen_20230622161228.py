'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2023-06-22 15:47:25
* LastEditors: LiuFeng
* LastEditTime: 2023-06-22 16:12:28
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

# validation function
def validate(
    model,
    dataset_valid,
    criterion,
    sequence_length,
    vocab_size,
    batch_size,
    device
):
    model.eval()
    print("#"*50)
    print("Validation")
    print("#"*50)
    
    valid_running_loss = 0.0
    counter = 0
    
    for i in tqdm(range(0,dataset_valid.size(0),sequence_length),total=int(dataset_valid.size(0)/sequence_length)):
        counter += 1
        inputs,labels = get_batch(
            dataset_valid,sequence_length,batch_size,device
        )
        
        # forward pass 
        with torch.no_grad():
            ouputs = model(inputs)
        
        # loss
        loss = criterion(
            ouputs.view(-1,vocab_size),
            labels.type(torch.int64)
        )
        valid_running_loss += loss.item()
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    
    return epoch_loss

class NLPDataset():
    def __init__(self,file_path,enc):
        self.file_path = file_path
        self.text_file = open(file_path)
        self.lines = self.text_file.read()
        self.enc = enc
    
    def __len__(self):
        return len(self.file_path)
    
    def get_data(self):
        final_vector = self.enc.encode(self.lines)
        return torch.tensor(final_vector[0::],dtype=torch.int32)