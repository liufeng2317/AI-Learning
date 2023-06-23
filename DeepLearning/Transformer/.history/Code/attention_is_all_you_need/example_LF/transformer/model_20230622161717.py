import torch 
import torch.nn as nn
import math
import torch.nn.functional as F 


class PositionalEncoding(nn.Module):
    def __init__(self,max_len,d_model,dropout=0.1):
        """
            :param max_len: Input length sequence.
            :param d_model: Embedding dimension.
            :param dropout: Dropout value (default=0.1)
        """
        super(PositionalEncoding,self).__init__()