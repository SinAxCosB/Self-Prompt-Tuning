import torch
import torch.nn as nn
from .MAE_vit import VisionTransformer
from functools import partial

__all__ = ['MAE_vpt_deep_vit_b', 'MAE_vpt_deep_vit_l', 'MAE_vpt_deep_vit_h']


class PromptVisionTransformer(VisionTransformer):
    def __init__(self, num_prompts=20, prompt_init=None, global_pool=True, **kwargs):
        super(PromptVisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool

        embed_dim = kwargs['embed_dim']
        self.depth = kwargs['depth']
        self.num_prompts = num_prompts
        
        assert self.depth == prompt_init.shape[0]
        # depth, num_prompts, embed_dim
        self.prompt_token = nn.Parameter(prompt_init, requires_grad=True)
    
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        prompt_tokens = self.prompt_token[0].unsqueeze(0).expand(B, -1, -1)
        x = torch.cat((
            x[:, :1, :],
            prompt_tokens,
            x[:, 1:, :]
            ), dim=1)
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if i == 0:
                x = blk(x)
            else:
                prompt_tokens = self.prompt_token[i].unsqueeze(0).expand(B, -1, -1)
                x = torch.cat((
                    x[:, :1, :],
                    prompt_tokens,
                    x[:, 1+self.num_prompts:, :]
                    ), dim=1)
                x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            out = self.norm(x)
        else:
            x = self.norm(x)
            out = x[:, 0]

        return out


def MAE_vpt_deep_vit_b(**kwargs):
    model = PromptVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def MAE_vpt_deep_vit_l(**kwargs):
    model = PromptVisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def MAE_vpt_deep_vit_h(**kwargs):
    model = PromptVisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model