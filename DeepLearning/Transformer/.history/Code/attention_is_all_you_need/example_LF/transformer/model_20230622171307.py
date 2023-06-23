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
        self.dropout = nn.Dropout(p=dropout)
        
        # pe ==> [sequence length, embed dim]
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
        
    def forward(self,x):
        """
        Inputs of forward function
        :param x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim = 512,n_heads = 8):
        """
        :param embed_dim: Embedding dimension.
        :param n_heads = Number of attention heads. 
        """
        super(MultiHeadAttention,self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        
        assert embed_dim%n_heads == 0,\
            f"Embedding dimension should be divisible by number of heads"
        
        self.head_dim = int(self.embed_dim/self.n_heads)
        
        # query matrix
        self.q = nn.Linear(self.head_dim,self.head_dim)
        # key matrix
        self.k = nn.Linear(self.head_dim,self.head_dim)
        # value amtrix
        self.v = nn.Linear(self.head_dim,self.head_dim)
        
        self.out = nn.Linear(self.n_heads*self.head_dim,self.embed_dim)
        
    def forward(self,key,query,value,mask=None):
        """
        :param key: key vector.
        :param query: query vector.
        :param value: value vector.
        :param mask: Whether masking or not, for decoder.
        """
        batch_size = key.size(0) # Batch size.
        seq_len = key.size(1) # Max. sequence length.
        inp_emb = key.size(2) # Embedding dim.
        assert inp_emb == self.embed_dim, \
            f"Input embedding {inp_emb} should match layer embedding {self.embed_dim}"
        
        seq_len_query = query.size(1)
        key = key.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ) # [bs, seq_len, n_heads, head_dim] ~ [32, 1024, 8, 64]
        query = query.view(
            batch_size, seq_len_query, self.n_heads, self.head_dim
        ) # [bs, seq_len, n_heads, head_dim] ~ [32, 1024, 8, 64]
        value = value.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ) # [bs, seq_len, n_heads, head_dim] ~ [32, 1024, 8, 64]
        
        k = self.k(key)
        q = self.q(query)
        v = self.v(value)
        
        k = k.transpose(1, 2) # [batch_size, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2) # [batch_size, n_heads, seq_len, head_dim]
        v = v.transpose(1, 2) # [batch_size, n_heads, seq_len, head_dim] 
        
        # scaled-dot product attention
        # Transeposed key for matrix multiplication.
        k_transposed = k.transpose(-1,-2)  # [batch_size, n_heads, head_dim, seq_len]
        dot = torch.matmul(q,k_transposed) # [batch_size, n_heads, seq_len, head_dim] * [batch_size, n_heads, head_dim, seq_len] = [batch_size, n_heads, seq_len, seq_len]
        if mask is not None:
            dot = dot.masked_fill_(mask == 0, float('-1e20'))
        
        # scaling
        dot = dot/math.sqrt(self.head_dim)
        scores = F.softmax(dot,dim = 1)
        
        # Dot product with value matrix
        scores = torch.matmul(scores,v) # [batch_size, n_heads, seq_len, seq_len] * [batch_size, n_heads, seq_len, head_dim] = [batch_size, n_heads, seq_len, head_dim]
        
        concat = scores.transpose(1,2).contiguous().view(
            batch_size,seq_len_query,self.head_dim*self.n_heads
        )
        
        out = self.out(concat)
        return out
        

class TransformerBlock(nn.Module):
    def __init__(self,embed_dim,expansion_factor=4,n_heads=8,dropout=0.3):
        """
            :param embed_dim: Embedding dimension.
            :param expansion_factor: Factor determining the output dimension
            of the linear layer.
            :param n_heads: Number of attention heads.
        """ 
        super(TransformerBlock,self).__init__()
