import torch
from base import BaseKVCache
class DynamicKVcache(BaseKVCache):
    def __init__(self,num_layers:int):
        self.keys:list[torch.Tensor | None] = [None for _ in range(num_layers)] 
        self.values:list[torch.Tensor | None] = [None for _ in range(num_layers)]
        self.num_layers = num_layers
        print(f"sucessfully initialize {num_layers} layers kv_cache")

    def update(self,
               layer_idx:int,
               k:torch.Tensor,
               v:torch.Tensor,
               ) -> tuple[torch.Tensor,torch.Tensor]:
        
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = k
            self.values[layer_idx] = v
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx],k],dim=-2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx],v],dim=-2)
        return self.keys[layer_idx],self.values[layer_idx]

    def get_seq_len(self, layer_idx):
        return int(self.keys[layer_idx].size(-2)) if self.keys[layer_idx] is not None else 0

    def reset(self):
        for i in range(self.num_layers):
            self.keys[i] = None
            self.values[i] = None
        print(f"finish reset")


if __name__ == "__main__":
    B,H,T,C = 1,4,128,8
    cache = DynamicKVcache(num_layers=2)

    #prefill T=5
    k_prefill = torch.randn(B,H,5,C)
    v_prefill = torch.randn(B,H,5,C)
    kc, kv =cache.update(layer_idx=0,k=k_prefill,v=v_prefill)
    print(f"prefill seq_len={cache.get_seq_len(0)}")

    #decode new T=1
    k_step = torch.randn(B,H,1,C)
    v_step = torch.randn(B,H,1,C)
    kc2,vc2 = cache.update(layer_idx=0,k=k_step,v=v_step)
    print(f"after decode seq_len={cache.get_seq_len(0)}")

    cache.reset()
    print(f"after reset seq_len={cache.get_seq_len(0)}")