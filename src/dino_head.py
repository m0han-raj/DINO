import torch
import torch.nn as nn

class DINOHead(nn.Module):
     """Projection MLP: 192 -> 512 -> 512 -> bottleneck 128 -> K prototypes.
    CPU edition: out_dim K = 4096 (paper uses 65536)."""
     def __init__(self,in_dim=192 ,hidden=512 , bottleneck=128 , out_dim=4096):
         super().__init__()
         self.mlp = nn.Sequential(
              nn.Linear(in_dim,hidden) , nn.GELU(),
              nn.Linear(hidden,hidden) , nn.GELU(),
              nn.Linear(hidden,bottleneck),
         )
         #weight normalized last layer , scale is frozen to 1
         self.last = nn.utils.weight_norm(
              nn.Linear(bottleneck,out_dim,bias =False)
         )
         self.last.weight_g.data.fill_(1)
         self.last.weight_g.requires_grad = False


     def forward(self,x):
       x=self.mlp(x)
       x = nn.functional.normalize(x,dim=1)
       return self.last(x)
     
