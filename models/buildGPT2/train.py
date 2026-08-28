import torch
from model.BigramLanguageModel import BigramLanguageModel

block_size = 256
batch_size = 32
epochs = 8000
eval_interval = 500
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embedd = 384*2
n_head = 12
n_layer = 18
drop_out =0.2

torch.manual_seed(1337)


with open("Shakespeare.txt",'r',encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s:[stoi[c] for c in s]
decode = lambda c:''.join(itos[i] for i in c)

def get_batch(split):
    data = train if split =="train" else val
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size]for i in ix])
    y = torch.stack([data[i+1:i+block_size+1]for i in ix])
    return x ,y

data = torch.tensor(encode(text), dtype=torch.long)
n=int(0.9*len(data))
train = data[:n]
val = data [n:]
vocab_size = len(chars)

m=BigramLanguageModel(vocab_size = vocab_size,n_embedd=n_embedd,block_size=block_size,
                      num_heads=n_head,drop_out=drop_out).to(device=device)

optim = torch.optim.AdamW(m.parameters(),lr =lr)

for steps in range(epochs):
    xb,yb = get_batch('train')
    xb =xb.to(device=device)
    yb =yb.to(device=device)
    logits,loss1 = m(xb,yb) 
    optim.zero_grad(set_to_none=True)
    loss1.backward()
    optim.step()

    if steps %100 ==0:
        xb,yb = get_batch('val')
        xb =xb.to(device=device)
        yb =yb.to(device=device)
        logits,loss = m(xb,yb)
        print(f"steps:{steps},train loss:{loss1.item()},val loss:{loss.item()}")
torch.save(m.state_dict(), f'model_epoch{epochs}_layers{n_layer}_n_embedd{n_embedd}.pt')
print(decode(m.generate(torch.zeros((1,1),dtype=torch.long,device=device),max_new_token=5000)[0].tolist()))