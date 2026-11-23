import torch
import torch.nn as nn


# patch embedding
class PatchEmbed(nn.Module):

    def __init__(self, img_size=64, patch_size=8, in_ch=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # 8x8=64
        self.proj = nn.Conv2d(
            in_ch, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (b,192,8,8)
        return x.flatten(2).transpose(1, 2)  # (b,64,192)


# Multi-head self attention
class Attention(nn.Module):
    def __init__(self, dim=192, num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,heads,N,head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if return_attention:
            return x, attn
        return x


# transformer block
class Block(nn.Module):
    def __init__(self, dim=192, num_heads=3, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, return_attention=False):
        if return_attention:
            y, attn = self.attn(self.norm1(x), return_attention=True)
            return x + y, attn
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ViT block
class VisionTransformer(nn.Module):
    def __init__(self, img_size=64, patch_size=8, embed_dim=192, depth=6, num_heads=3):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _interp_pos(self, x):
        """Interpolate pos embeddings when input size differs (32 vs 64)."""
        n_x, n_p = x.shape[1] - 1, self.pos_embed.shape[1] - 1
        if n_x == n_p:
            return self.pos_embed
        cls_pe, patch_pe = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        dim = x.shape[-1]
        s_old, s_new = int(n_p**0.5), int(n_x**0.5)
        patch_pe = patch_pe.reshape(1, s_old, s_old, dim).permute(0, 3, 1, 2)
        patch_pe = nn.functional.interpolate(
            patch_pe, size=(s_new, s_new), mode="bicubic"
        )
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat([cls_pe, patch_pe], dim=1)

    def forward(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self._interp_pos(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token = image representation

    def get_last_attention(self, x):
        """For Part 5 visualization."""
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self._interp_pos(x)
        for blk in self.blocks[:-1]:
            x = blk(x)
        _, attn = self.blocks[-1](x, return_attention=True)
        return attn  # (B, heads, N+1, N+1)
