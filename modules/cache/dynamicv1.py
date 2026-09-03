import torch
from .base import BaseKVCache
from typing import Optional

class DynamicKVCacheV1(BaseKVCache):
    def __init__(self,
                 num_layers:int,
                 block_size:int,
                 device: Optional[torch.device] = None,
                 dtype:torch.dtype = torch.float32,
                 ):
        self.num_layers = num_layers
        self.block_size = block_size
        self.device = device
        self.dtype = dtype

        self.keys: list[torch.Tensor | None] = [None for _ in range(num_layers)]
        self.values: list[torch.Tensor | None] = [None for _ in range(num_layers)]
        self.cur_pos: list[int] = [0 for _ in range(num_layers)]  # 每层独立pos

    def _alloc_block(self,B:int,H:int,C:int,)->torch.Tensor:
         return torch.zeros(
              (B,H,self.block_size,C),
              device=self.device,
              dtype=self.dtype,
         )
    def update(self,
               layer_idx:int,
               k:torch.Tensor,
               v:torch.Tensor,
               )->tuple[torch.Tensor,torch.Tensor]:
         """
         k,v:[B,H,T_new,C]
         return:(k_cache,v_cache) ([B,H,T,C],[B,H,T,C])
         """
         B,H,T_new,C = k.size()
         pos = self.cur_pos[layer_idx]
         if self.keys[layer_idx] is None:
            k_buff = self._alloc_block(B,H,C)
            v_buff = self._alloc_block(B,H,C)
         else:
              k_buff = self.keys[layer_idx]
              v_buff = self.values[layer_idx]

         buff_len = k_buff.size(-2)

         while pos+T_new>buff_len:
              new_k_buff = self._alloc_block(B,H,C)
              new_v_buff = self._alloc_block(B,H,C)
              k_buff = torch.cat([k_buff,new_k_buff],dim=-2)
              v_buff = torch.cat([v_buff,new_v_buff],dim=-2)

         k_buff[:,:,pos:pos+T_new,:] = k
         v_buff[:,:,pos:pos+T_new,:] = v

         self.keys[layer_idx] = k_buff
         self.values[layer_idx] = v_buff
         self.cur_pos[layer_idx] += T_new
         valid_len = self.cur_pos[layer_idx]
         return k_buff[:,:,:valid_len,:],v_buff[:,:,:valid_len,:]
    
    def get_seq_len(self, layer_idx):
            return self.cur_pos[layer_idx] if self.keys[layer_idx] is not None else 0

    def reset(self):
        print(f"Now start reseting...")
        self.keys = [None for _ in range(self.num_layers)]
        self.values = [None for _ in range(self.num_layers)]
        self.cur_pos = [0 for _ in range(self.num_layers)]
        print(f"Finish")

if __name__ == "__main__":
    B, H_KV, C = 1,4,128
    cache = DynamicKVCacheV1(num_layers=2, block_size=10)

    # prefill T=5
    k_prefill = torch.randn(B,H_KV,5,C)
    v_prefill = torch.randn(B,H_KV,5,C)
    kc,vc = cache.update(0, k_prefill, v_prefill)
    print(f"layer0 seq_len={cache.get_seq_len(0)}") #5

    # decode step *3
    for _ in range(3):
        ks = torch.randn(B,H_KV,1,C)
        vs = torch.randn(B,H_KV,1,C)
        kc,vc = cache.update(0, ks, vs)
    print(f"layer0 seq_len={cache.get_seq_len(0)}") #8

    cache.reset()
    print(f"after reset seq_len={cache.get_seq_len(0)}") #0
