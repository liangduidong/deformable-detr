import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """前馈神经网络 (FFN)"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
if __name__ == '__main__':
    x = torch.rand(2, 10, 512)  # [batch, seq_len, d_model]
    ffn = FeedForward(512, 2048)
    x = ffn(x)
    print(x.shape)