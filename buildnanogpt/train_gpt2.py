import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import math
import tiktoken
import inspect
import time
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ---------------------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257 # 扩大到50304 （2的幂次）后效率更高，适应了cuda内核
    n_layer: int = 12
    n_head: int = 12
    n_embd:int = 768

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,config.n_embd *4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4* config.n_embd,config.n_embd)

        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self,x):
        x = self.gelu(self.c_fc(x))
        return self.c_proj(x)

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 并不是bias ， only  follow openAI/HF naming through
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1,1,config.block_size,config.block_size))
        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self,x):
        B,T,C = x.shape #(B,T,C)
        qkv = self.c_attn(x) #B,T,3C
        q ,k ,v= qkv.split(self.n_embd,dim=2) #3(B,T,C)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # B,nh,T,c/nh

        # att = (q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1))) # B nh T T
        # att = att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        # att = F.softmax(att,dim=-1)
        # y = att @ v #B nh T nh
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True) # 要显式调用flashattn

        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y
    
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x+self.attn(self.ln_1(x))
        x = x+self.mlp(self.ln_2(x))
        return x
    

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config =config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            std = 0.02
            if hasattr(module,"NANOGPT_SCALE_INIT"):
                std +=(2*self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std =std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,idx,target = None):
        B,T = idx.size()
        assert T<=self.config.block_size,f"不能前向传播{T}这么多token"
        pos = torch.arange(0,T,dtype=torch.long,device=device)
        pos_embd = self.transformer.wpe(pos)
        token_embd = self.transformer.wte(idx)
        x = pos_embd + token_embd
        for b in self.transformer.h:
            x = b(x)
        x = self.transformer.ln_f(x)
        logit=self.lm_head(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logit.view(-1,logit.size(-1)),target.view(-1))
        return logit , loss

    def configure_optimizer(self,weight_decay = 0.1,learning_rate = 6e-4,device=device):
        param_dict = {pn:p for pn,p in self.named_parameters()}
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}

        decay_param = [p for n ,p in param_dict.items() if p.dim() >=2]
        nodecay_param = [p for n ,p in param_dict.items() if p.dim() <2]
        optim_groups = [
            {'params':decay_param,'weight_decay':weight_decay},
            {'params':nodecay_param,'weight_decay':0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_param)
        num_nodecay_params = sum(p.numel() for p in nodecay_param)
        print(f"num decayed parameter tensors:{len(decay_param)},with {num_decay_params:,} param")
        print(f"num no-decayed parameter tensors:{len(nodecay_param)},with {num_nodecay_params:,} param")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters # 内核融合的方案
        used_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW :{used_fused}")
        optimizer = torch.optim.AdamW(optim_groups,lr=learning_rate,betas=(0.9,0.95),eps=1e-8,fused=used_fused)
        return optimizer
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt,dtype= torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self,B,T,process_rank,num_processes,split):
        self.B = B 
        self.T = T 
        self.process_rank=process_rank
        self.num_processes = num_processes
        assert split in {'train','val'}

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root,s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for {split}"
        if master_process :
            print(f"found {len(shards)} shards for {split}")

        self.reset()

    def reset(self):
        #state
        self.current_shard =0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B*self.T*self.process_rank

    def next_batch(self):
        B,T = self.B,self.T 
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        self.current_position+=B*T*self.num_processes
        if self.current_position + (B*T*self.num_processes+1) >len(self.tokens):
            self.current_shard = (self.current_shard + 1)%len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B*self.T*self.process_rank
        return x,y

# --------------------------------------------------------------------------------------
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")        
# ------------------------------------------------------------------------------------------------
def get_lr(it): #带预热的学习率余弦衰减
    if it <warmup_step:
        return max_lr *(it+1)/warmup_step
    
    if it >max_step:
        return min_lr
    decay_ratio = (it-warmup_step)/(max_step - warmup_step)
    assert 0<=decay_ratio<=1
    coeff = 0.5*(1.0+math.cos(math.pi*decay_ratio))
    return min_lr+coeff*(max_lr-min_lr)
# ------------------------------------------------------

max_lr = 6e-4
min_lr = max_lr*0.1
total_batch_size = 524288
B = 4
T=1024
max_step = 19073
warmup_step = 715

max_length = 32
max_return_sequence = 4
enc = tiktoken.get_encoding('gpt2') # val 
# ---------------------------------------------------
# 梯度累计
assert total_batch_size%(B*T)==0
grad_accum_steps = total_batch_size//(B*T*ddp_world_size)
if master_process:
    print(f"total desired batch size:{total_batch_size}")
    print(f"gradient need {grad_accum_steps} steps to accumlate")
train_loader = DataLoaderLite(B=B,T=T,process_rank=ddp_rank,num_processes=ddp_world_size,split='train')
val_loader = DataLoaderLite(B=B,T=T,process_rank=ddp_rank,num_processes=ddp_world_size,split='val')
#----------------------------------------------------------------
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

torch.set_float32_matmul_precision('medium')

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model) # 编译，不使用python解释器，一种内核融合，但是Windows不行

if ddp:
    model = DDP(model,device_ids=[ddp_local_rank])
raw_model =model.module if ddp else model

optimizer =raw_model.configure_optimizer(weight_decay = 0.1,learning_rate = 6e-4,device=device)

# 训练日志log
log_dir = "log"
os.makedirs(log_dir,exist_ok=True)
log_file = os.path.join(log_dir,"log.txt")
with open(log_file ,"w") as f:
    pass

for i in range(max_step):
    if i%100 ==0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0
            val_loss_step = 20
            for _ in range(val_loss_step):
                val_x,val_y = val_loader.next_batch()
                val_x,val_y = val_x.to(device),val_y.to(device)
                with torch.autocast(device,dtype=torch.bfloat16):
                    _,val_loss = model(val_x,val_y)
                val_loss/=val_loss_step
                val_loss_accum +=val_loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum,op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss :{val_loss_accum.item():.4f}")
            f.write(f"validation loss :{val_loss_accum.item():.4f}")

        # 测试输出样例
        model.eval()
        prompt = enc.encode("Hello l'm a language model,")
        prompt = torch.tensor(prompt,dtype=torch.long)
        prompt = prompt.unsqueeze(0).repeat(max_return_sequence,1)
        x = prompt.to(device=device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42+ddp_rank)
        while x.size(1)<max_length:
            with torch.no_grad():
                logits = model(x)
                logits = logits[:,-1,:]
                probs = F.softmax(logits,dim=-1)
                topk_probs ,topk_indices = torch.topk(probs,50,dim=-1)
                ix = torch.multinomial(topk_probs,1)
                xcol = torch.gather(topk_indices,-1,ix)
                x = torch.cat((x,xcol),dim=1)

        for i in range(max_return_sequence):
            tokens = x[i,:max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} steps: {i} sample",decoded)
            

    model.train()
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0
    for mircostep in range(grad_accum_steps): # 累计梯度
        x,y = train_loader.next_batch()
        x,y = x.to(device=device),y.to(device)
        with torch.autocast(device_type=device,dtype=torch.bfloat16): # 混合精度 参数是float32，部分计算是bf16.
            logits,loss = model(x,y)
        loss/=grad_accum_steps
        loss_accum+=loss.detach()
        if ddp:
            model.require_backward_grad_sync = (mircostep == grad_accum_steps -1) #在积累完梯度后通信梯度
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum,op =dist.ReduceOp.AVG) # 规约总损失
    norm = torch.nn.utils.clip_grad_norm(model.parameters(),max_norm=1.0) # 限制梯度大小
    torch.cuda.synchronize() #cpu gpu 同步以计时
    t1 = time.time()
    lr = get_lr(i)
    for modelp in optimizer.param_groups:
        modelp['lr'] = lr
    optimizer.step()
    if master_process:
        print(f"step{i},loss:{loss_accum:.4f} | dt:{(t1-t0)*1000:.4f} | token/sec: \
    {train_loader.B*train_loader.T/(t1-t0):.4f} | gradient norm:{norm:.4f} | lr:{lr:.4f}")
if ddp:
    destroy_process_group()

