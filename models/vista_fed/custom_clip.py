from dataclasses import dataclass
from typing import Optional, Tuple, Union
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .eva_vit_model import EVAVisionTransformer
from .transformer import LayerNorm, QuickGELU

FusedLayerNorm = LayerNorm

@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0. # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    global_average_pool: bool = False # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    drop_path_rate: Optional[float] = None  # drop path rate
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    eva_model_name: str = None # a valid eva model name overrides layers, width, patch_size
    qkv_bias: bool = True
    fusedLN: bool = False
    xattn: bool = False
    postnorm: bool = False
    rope: bool = False
    pt_hw_seq_len: int = 16   # 224/14
    intp_freq: bool = False
    naiveswiglu: bool = False
    subln: bool = False


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    masked_language_modeling: bool = False
    fusedLN: bool = False
    xattn: bool = False
    attn_mask: bool = True

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    vision_heads = vision_cfg.width // vision_cfg.head_width
    norm_layer = LayerNorm
    
    visual = EVAVisionTransformer(
        img_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        num_classes=embed_dim,
        use_mean_pooling=vision_cfg.global_average_pool, #False
        init_values=vision_cfg.ls_init_value,
        patch_dropout=vision_cfg.patch_dropout,
        embed_dim=vision_cfg.width,
        depth=vision_cfg.layers,
        num_heads=vision_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        qkv_bias=vision_cfg.qkv_bias,
        drop_path_rate=vision_cfg.drop_path_rate,
        norm_layer= partial(FusedLayerNorm, eps=1e-6) if vision_cfg.fusedLN else partial(norm_layer, eps=1e-6),
        xattn=vision_cfg.xattn,
        rope=vision_cfg.rope,
        postnorm=vision_cfg.postnorm,
        pt_hw_seq_len= vision_cfg.pt_hw_seq_len,   # 224/14
        intp_freq= vision_cfg.intp_freq,
        naiveswiglu= vision_cfg.naiveswiglu,
        subln= vision_cfg.subln
    )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    return None


class CustomCLIP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            itm_task: bool = False,
            is_only_visual: bool = False,
            is_only_text: bool = False,
    ):
        super().__init__()
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) #可学习参数
        if is_only_visual:
            self.text = None
        if is_only_text:
            self.visual = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers:int=0, freeze_layer_norm:bool=True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        if self.text is not None:
            self.text.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'logit_scale'}

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text):
        if self.visual is not None:
            image_features = self.encode_image(image, normalize=True)
        else:
            image_features = None
        if self.text is not None:
            text_features = self.encode_text(text, normalize=True)
        else:
            text_features = None
        return image_features, text_features, self.logit_scale.exp()