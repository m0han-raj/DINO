import torch
import torch.nn as nn

#patch embedding
class PatchEmbed(nn.Module):

    def __init__(self,img_size = 64 , patch_size = 8 , in_ch =3,embed_dim=192):
        super().__init__()
        self.num_patches = (img_size//patch_size)**2 #8x8=64
        self.proj = nn.Conv2d(in_ch,embed_dim,kernel_size=patch_size,stride=patch_size)
    
    def forward(self,x):
        x = self.proj(x) #(b,192,8,8)
        return x.flatten(2).transpose(1,2) #(b,64,192)
    
#Multi-head self attention
class Attention(nn.Module):
    def __init__(self,dim=192,num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim//num_heads) ** -0.5
        self.qkv = nn.Linear(dim,dim*3)
        self.proj = nn.Linear(dim,dim)

    def forward(self,x,return_attention=False):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads , C//self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv[0] , qkv[1] , qkv[2] #(B,heads,N,head_dim)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=1)
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        if return_attention:
            return x,attn
        return x
    
#transformer block
class Block(nn.Module):
    def __init__(self, dim=192, num_heads=3, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x, return_attention=False):
        if return_attention:
            y, attn = self.attn(self.norm1(x), return_attention=True)
            return x + y, attn
        x = x + self.attn(self.norm1(x))       
        x = x + self.mlp(self.norm2(x))        
        return x