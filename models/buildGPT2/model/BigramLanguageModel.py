import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
    def __init__(self,block_size,n_embedd,headdim = 16):
        super().__init__()
        self.q = nn.Linear(n_embedd,headdim)
        self.k = nn.Linear(n_embedd,headdim)
        self.v = nn.Linear(n_embedd,headdim)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C = x.shape # (B,T,C)
        wei = self.q(x)@self.k(x).transpose(-2,-1)*C**-0.5 #(B,T,head_dim) @(B,head_dim,T) -->(B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0,float('-inf'))
        wei = F.softmax(wei,dim=-1)
        v = self.v(x) #(B,T,head_dim)
        out = wei@v #(B,T,T)@(B,T,head_dim) --> (B,T,head_dim)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,block_size,n_embedd,num_heads,headdim = 16,drop = 0.3):
        super().__init__()
        self.multi = nn.ModuleList([Head(block_size,n_embedd,headdim =headdim)for _ in range(num_heads)])
        self.pro = nn.Linear(n_embedd,n_embedd)
        self.drop = nn.Dropout(drop)
    def forward(self,x):
        out = torch.cat([h(x)for h in self.multi],dim=-1)
        out = self.pro(out)
        return self.drop(out)

class FeedForward(nn.Module):
    def __init__(self,n_embedd,drop):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(n_embedd,4*n_embedd),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        self.l2 = nn.Linear(4*n_embedd,n_embedd)
    def forward(self,x):
        return self.l2(self.l1(x))

class LayerNorm(nn.Module):
    def __init__(self, dim,eps =1e-5):
        super().__init__()
        self.eps =eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))

    def forward(self,x):
        xmean = x.mean(dim=-1, keepdim=True)
        xvar = x.var(dim=-1, keepdim=True, unbiased=False)
        xhat = (x-xmean)/torch.sqrt(xvar+self.eps)
        self.out = self.gamma*xhat + self.beta
        return self.out
    
class Block(nn.Module):
    def __init__(self,block_size,n_embedd,n_heads,drop):
        super().__init__()
        head_dim = n_embedd // n_heads
        self.a = MultiHeadAttention(block_size,n_embedd,n_heads,headdim =head_dim,drop=drop)
        self.l = FeedForward(n_embedd=n_embedd,drop=drop)
        self.ln1 = LayerNorm(n_embedd)
        self.ln2 = LayerNorm(n_embedd)
    def forward(self,x):
        x = x + self.a(self.ln1(x))
        x = x + self.l(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size,n_embedd,block_size=32,num_heads=4,drop_out = 0.5):
        super().__init__()
        self.token_embedding_table =nn.Embedding(vocab_size,n_embedd)
        self.lm_head = nn.Linear(n_embedd,vocab_size)
        self.position_embedding_table = nn.Embedding(block_size,n_embedd)
        self.blocks = nn.Sequential(
            Block(n_embedd=n_embedd,block_size=block_size,n_heads=num_heads,drop=drop_out),
            Block(n_embedd=n_embedd,block_size=block_size,n_heads=num_heads,drop=drop_out),
            Block(n_embedd=n_embedd,block_size=block_size,n_heads=num_heads,drop=drop_out),
            Block(n_embedd=n_embedd,block_size=block_size,n_heads=num_heads,drop=drop_out),
            Block(n_embedd=n_embedd,block_size=block_size,n_heads=num_heads,drop=drop_out),
            LayerNorm(n_embedd)
        )
        self.block_size =block_size

    def forward(self,idx,target=None):
        B,T = idx.shape

        logits = self.token_embedding_table(idx)
        pos_embedd = self.position_embedding_table(torch.arange(T,device=device))
        x= logits + pos_embedd
        x = self.blocks(x)
        logits = self.lm_head(x)
        B,T,C = logits.shape
        if target is None:
            loss =None
        else:    
            logits = logits.view(-1,C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits,target)
        return logits,loss

    def generate(self,idx,max_new_token):
        for _ in range(max_new_token):
            idx_cond = idx[:,-self.block_size:]
            logits , loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx