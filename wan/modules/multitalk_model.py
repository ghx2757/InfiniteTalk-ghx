# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import numpy as np
import os
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .attention import flash_attention, SingleStreamMutiAttention
from ..utils.multitalk_utils import get_attn_map_with_target
import logging
try:
    from sageattention import sageattn
    USE_SAGEATTN = True
    logging.info("Using sageattn")
except:
    USE_SAGEATTN = False

__all__ = ['WanModel']



def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):

    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    s, n, c = x.size(1), x.size(2), x.size(3) // 2

    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)
        freqs_i = freqs_i.to(device=x_i.device)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        out = F.layer_norm(
            inputs.float(), 
            self.normalized_shape, 
            None if self.weight is None else self.weight.float(), 
            None if self.bias is None else self.bias.float() ,
            self.eps
        ).to(origin_dtype)
        return out


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, ref_target_masks=None):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v
        q, k, v = qkv_fn(x)

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        if USE_SAGEATTN:
            x = sageattn(q.to(torch.bfloat16), k.to(torch.bfloat16), v, tensor_layout='NHD')
        else:
            x = flash_attention(
                q=q,
                k=k,
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size
            ).type_as(x)

        # output
        x = x.flatten(2)
        x = self.o(x)
        with torch.no_grad():
            x_ref_attn_map = get_attn_map_with_target(q.type_as(x), k.type_as(x), grid_sizes[0], 
                                                    ref_target_masks=ref_target_masks)

        return x, x_ref_attn_map


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        if USE_SAGEATTN:
            img_x = sageattn(q, k_img, v_img, tensor_layout='NHD')
            x = sageattn(q, k, v, tensor_layout='NHD')
        else:   
            img_x = flash_attention(q, k_img, v_img, k_lens=None)
            # compute attention
            x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 output_dim=768,
                 norm_input_visual=True,
                 class_range=24,
                 class_interval=4):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanI2VCrossAttention(dim,
                                                num_heads,
                                                (-1, -1),
                                                qk_norm,
                                                eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # init audio module
        self.audio_cross_attn = SingleStreamMutiAttention(
                dim=dim,
                encoder_hidden_states_dim=output_dim,
                num_heads=num_heads,
                qk_norm=False,
                qkv_bias=True,
                eps=eps,
                norm_layer=WanRMSNorm,
                class_range=class_range,
                class_interval=class_interval
            )
        self.norm_x = WanLayerNorm(dim, eps, elementwise_affine=True)  if norm_input_visual else nn.Identity()
        

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        audio_embedding=None,
        ref_target_masks=None,
        human_num=None,
    ):

        dtype = x.dtype
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation.to(e.device) + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y, x_ref_attn_map = self.self_attn(
            (self.norm1(x).float() * (1 + e[1]) + e[0]).type_as(x), seq_lens, grid_sizes,
            freqs, ref_target_masks=ref_target_masks)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]
        
        x = x.to(dtype)

        # cross-attention of text
        x = x + self.cross_attn(self.norm3(x), context, context_lens)

        # cross attn of audio
        x_a = self.audio_cross_attn(self.norm_x(x), encoder_hidden_states=audio_embedding,
                                        shape=grid_sizes[0], x_ref_attn_map=x_ref_attn_map, human_num=human_num)
        x = x + x_a

        y = self.ffn((self.norm2(x).float() * (1 + e[4]) + e[3]).to(dtype))
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]


        x = x.to(dtype)

        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation.to(e.device) + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class AudioProjModel(ModelMixin, ConfigMixin):
    def __init__(
        self,
        seq_len=5,
        seq_len_vf=12,
        blocks=12,  
        channels=768, 
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        norm_output_audio=False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels  
        self.input_dim_vf = seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj1_vf = nn.Linear(self.input_dim_vf, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def forward(self, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B, _, _, S, C = audio_embeds.shape

        # process audio of first frame
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        # process audio of latter frame
        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) w b c")
        batch_size_vf, window_size_vf, blocks_vf, channels_vf = audio_embeds_vf.shape
        audio_embeds_vf = audio_embeds_vf.view(batch_size_vf, window_size_vf * blocks_vf * channels_vf)

        # first projection
        audio_embeds = torch.relu(self.proj1(audio_embeds)) 
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf)) 
        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=B)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=B)
        audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1) 
        batch_size_c, N_t, C_a = audio_embeds_c.shape
        audio_embeds_c = audio_embeds_c.view(batch_size_c*N_t, C_a)

        # second projection
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))

        context_tokens = self.proj3(audio_embeds_c).reshape(batch_size_c*N_t, self.context_tokens, self.output_dim)

        # normalization and reshape
        with amp.autocast(dtype=torch.float32):
            context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)

        return context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
class WanModel(ModelMixin, ConfigMixin):
    # ============================================================
    # __init__ 函数逻辑梳理:
    # ============================================================
    # 输入: 模型配置参数
    # │
    # ├─ 1. 模型类型检查与基础参数设置
    # │   └─ assert model_type == 'i2v' → 保存所有配置参数
    # │
    # ├─ 2. Embeddings 层初始化
    # │   ├─ patch_embedding: Conv3d → 视频patch嵌入
    # │   ├─ text_embedding: Linear→GELU→Linear → 文本嵌入
    # │   └─ time_embedding: Linear→SiLU→Linear → 时间步嵌入
    # │       └─ time_projection: SiLU→Linear(6*dim) → 时间调制投影
    # │
    # ├─ 3. Attention Blocks 初始化
    # │   └─ 创建 num_layers 个 WanAttentionBlock
    # │       ├─ 包含 Self-Attention
    # │       ├─ 包含 Cross-Attention (文本)
    # │       ├─ 包含 Audio Cross-Attention
    # │       └─ 包含 FFN
    # │
    # ├─ 4. 输出 Head 初始化
    # │   └─ Head(dim, out_dim, patch_size) → unpatchify输出层
    # │
    # ├─ 5. RoPE 频率参数初始化
    # │   └─ freqs = [rope_params×3] → 时空位置编码
    # │
    # ├─ 6. 图像投影模块 (i2v模式)
    # │   └─ img_emb: MLPProj(1280→dim) → CLIP特征投影
    # │
    # ├─ 7. 音频投影模块初始化
    # │   └─ audio_proj: AudioProjModel → 音频特征处理
    # │       ├─ seq_len: audio_window
    # │       ├─ seq_len_vf: audio_window + vae_scale - 1
    # │       └─ output_dim, context_tokens, norm设置
    # │
    # └─ 8. 权重初始化
    #     └─ weight_init=True → init_weights()
    #
    # ============================================================
    
    @register_to_config
    def __init__(self,
                 model_type='i2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 # audio params
                 audio_window=5,
                 intermediate_dim=512,
                 output_dim=768,
                 context_tokens=32,
                 vae_scale=4, # vae timedownsample scale

                 norm_input_visual=True,
                 norm_output_audio=True,
                 weight_init=True):
        super().__init__()

        assert model_type == 'i2v', 'MultiTalk model requires your model_type is i2v.'
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps


        self.norm_output_audio = norm_output_audio
        self.audio_window = audio_window
        self.intermediate_dim = intermediate_dim
        self.vae_scale = vae_scale
        

        # embeddings
        self.patch_embedding = nn.Conv3d( # 【patch_embedding】
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential( # 【text_embedding】
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential( # 【时间步嵌入】
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6)) # 【时间调制投影】

        # blocks
        cross_attn_type = 'i2v_cross_attn'
        self.blocks = nn.ModuleList([ # 【Attention Blocks】
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, 
                              output_dim=output_dim, norm_input_visual=norm_input_visual)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps) #【输出 Head】

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads # 【 RoPE 频率参数初始化】
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        if model_type == 'i2v': 
            self.img_emb = MLPProj(1280, dim) # 【CLIP特征投影】
        else:
            raise NotImplementedError('Not supported model type.')
        
        # init audio adapter
        self.audio_proj = AudioProjModel( # 【音频投影】
                    seq_len=audio_window,
                    seq_len_vf=audio_window+vae_scale-1,
                    intermediate_dim=intermediate_dim,
                    output_dim=output_dim,
                    context_tokens=context_tokens,
                    norm_output_audio=norm_output_audio,
                )


        # initialize weights
        if weight_init:
            self.init_weights()
            
    def init_freqs(self):
        d = self.dim // self.num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

    def teacache_init(
        self,
        use_ret_steps=True,
        teacache_thresh=0.2,
        sample_steps=40,
        model_scale='infinitetalk-480',
    ):
        print("teacache_init")
        self.enable_teacache = True
        
        self.__class__.cnt = 0
        self.__class__.num_steps = sample_steps*3
        self.__class__.teacache_thresh = teacache_thresh
        self.__class__.accumulated_rel_l1_distance_even = 0
        self.__class__.accumulated_rel_l1_distance_odd = 0
        self.__class__.previous_e0_even = None
        self.__class__.previous_e0_odd = None
        self.__class__.previous_residual_even = None
        self.__class__.previous_residual_odd = None
        self.__class__.use_ret_steps = use_ret_steps

        if use_ret_steps:
            if model_scale == 'infinitetalk-480':
                self.__class__.coefficients = [ 2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01]
            if model_scale == 'infinitetalk-720':
                self.__class__.coefficients = [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02]
            self.__class__.ret_steps = 5*3
            self.__class__.cutoff_steps = sample_steps*3
        else:
            if model_scale == 'infinitetalk-480':
                self.__class__.coefficients = [-3.02331670e+02,  2.23948934e+02, -5.25463970e+01,  5.87348440e+00, -2.01973289e-01]
        
            if model_scale == 'infinitetalk-720':
                self.__class__.coefficients = [-114.36346466,   65.26524496,  -18.82220707,    4.91518089,   -0.23412683]
            self.__class__.ret_steps = 1*3
            self.__class__.cutoff_steps = sample_steps*3 - 3
        print("teacache_init done")
    
    def disable_teacache(self):
        self.enable_teacache = False
# forward 函数逻辑梳理:
    # ============================================================
    # 输入: x(视频latent), t(时间步), context(文本), clip_fea, y(引导), audio, ref_target_masks
    # │
    # ├─ 1. 输入预处理与尺寸计算
    # │   ├─ 获取 T, H, W → 计算 patch后的 N_t, N_h, N_w
    # │   └─ y存在? → x = [cat(x[i], y[i])] (引导合并)
    # │
    # ├─ 2. Patch Embedding处理
    # │   ├─ x → patch_embedding(Conv3d)
    # │   ├─ 记录 grid_sizes, seq_lens
    # │   └─ flatten + padding到seq_len
    # │
    # ├─ 3. 时间Embedding
    # │   ├─ t → sinusoidal_embedding_1d → time_embedding
    # │   └─ e → time_projection → e0 [B, 6, dim] (调制参数)
    # │
    # ├─ 4. 文本Embedding
    # │   └─ context → text_embedding + padding到text_len
    # │
    # ├─ 5. CLIP特征融合
    # │   └─ clip_fea → img_emb → concat到context前面
    # │
    # ├─ 6. 音频特征处理
    # │   ├─ audio分离: first_frame + latter_frame
    # │   ├─ latter_frame重排: (b, n_t*vae_scale, w, s, c) → (b, n_t, vae_scale, w, s, c)
    # │   ├─ 按时间窗口提取:
    # │   │   ├─ 首帧: [:middle_index+1] → latter_first_frame_audio_emb
    # │   │   ├─ 中间帧: [middle_index:middle_index+1] → latter_middle_frame_audio_emb
    # │   │   └─ 末帧: [middle_index:] → latter_last_frame_audio_emb
    # │   ├─ concat三部分 → latter_frame_audio_emb_s
    # │   └─ audio_proj(first_frame, latter_frame) → audio_embedding [human_num, f, m, c]
    # │
    # ├─ 7. 参考目标Masks处理
    # │   └─ ref_target_masks存在?
    # │       └─ interpolate到(N_h, N_w) → token级别masks
    # │
    # ├─ 8. TeaCache加速逻辑 (可选)
    # │   │
    # │   ├─ enable_teacache=True?
    # │   │   ├─ 根据 cnt%3 区分: cond(0) / drop_text(1) / uncond(2)
    # │   │   ├─ 计算 accumulated_rel_l1_distance (累积L1距离)
    # │   │   ├─ distance < thresh?
    # │   │   │   ├─ 是 → should_calc=False (重用缓存)
    # │   │   │   └─ 否 → should_calc=True (重新计算)
    # │   │   └─ 保存 previous_e0, previous_residual
    # │   │
    # │   └─ enable_teacache=False?
    # │       └─ 正常执行所有blocks
    # │
    # ├─ 9. Attention Blocks处理
    # │   ├─ 构建kwargs参数包 (e0, seq_lens, grid_sizes, freqs, context, audio_embedding, masks, human_num)
    # │   │
    # │   └─ TeaCache启用?
    # │       ├─ 是 → 根据should_calc决定:
    # │       │   ├─ should_calc=True → 正常执行blocks, 保存residual
    # │       │   └─ should_calc=False → x += previous_residual (跳过计算)
    # │       │
    # │       └─ 否 → for block in blocks: x = block(x, **kwargs)
    # │
    # ├─ 10. Head输出层
    # │   └─ head(x, e) → patch级输出
    # │
    # ├─ 11. Unpatchify恢复
    # │   └─ unpatchify(x, grid_sizes) → 恢复视频tensor形状
    # │
    # └─ 12. TeaCache计数更新
    #     └─ enable_teacache → cnt+=1, 到num_steps时重置
    #
    # 输出: List[Tensor] → [C_out, F, H, W] 视频latent
    # ============================================================
    def forward(
            self,
            x,
            t,
            context,
            seq_len,
            clip_fea=None,
            y=None,
            audio=None,
            ref_target_masks=None,
        ):
        assert clip_fea is not None and y is not None

        _, T, H, W = x[0].shape
        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        x[0] = x[0].to(context[0].dtype)

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # text embedding
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # clip embedding
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea) 
            context = torch.concat([context_clip, context], dim=1).to(x.dtype)

        
        audio_cond = audio.to(device=x.device, dtype=x.dtype)
        first_frame_audio_emb_s = audio_cond[:, :1, ...] 
        latter_frame_audio_emb = audio_cond[:, 1:, ...] 
        latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=self.vae_scale) 
        middle_index = self.audio_window // 2
        latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index+1, ...] 
        latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
        latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...] 
        latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
        latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index+1, ...] 
        latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
        latter_frame_audio_emb_s = torch.concat([latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2) 
        audio_embedding = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s) 
        human_num = len(audio_embedding)
        audio_embedding = torch.concat(audio_embedding.split(1), dim=2).to(x.dtype)


        # convert ref_target_masks to token_ref_target_masks
        if ref_target_masks is not None:
            ref_target_masks = ref_target_masks.unsqueeze(0).to(torch.float32) 
            token_ref_target_masks = nn.functional.interpolate(ref_target_masks, size=(N_h, N_w), mode='nearest') 
            token_ref_target_masks = token_ref_target_masks.squeeze(0)
            token_ref_target_masks = (token_ref_target_masks > 0)
            token_ref_target_masks = token_ref_target_masks.view(token_ref_target_masks.shape[0], -1) 
            token_ref_target_masks = token_ref_target_masks.to(x.dtype)

        # teacache
        if self.enable_teacache:
            modulated_inp = e0 if self.use_ret_steps else e
            if self.cnt%3==0: # cond
                if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_cond = True
                    self.accumulated_rel_l1_distance_cond = 0
                else:
                    rescale_func = np.poly1d(self.coefficients)
                    self.accumulated_rel_l1_distance_cond += rescale_func(((modulated_inp-self.previous_e0_cond).abs().mean() / self.previous_e0_cond.abs().mean()).cpu().item())
                    if self.accumulated_rel_l1_distance_cond < self.teacache_thresh:
                        should_calc_cond = False
                    else:
                        should_calc_cond = True
                        self.accumulated_rel_l1_distance_cond = 0
                self.previous_e0_cond = modulated_inp.clone()
            elif self.cnt%3==1: # drop_text
                if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_drop_text = True
                    self.accumulated_rel_l1_distance_drop_text = 0
                else:
                    rescale_func = np.poly1d(self.coefficients)
                    self.accumulated_rel_l1_distance_drop_text += rescale_func(((modulated_inp-self.previous_e0_drop_text).abs().mean() / self.previous_e0_drop_text.abs().mean()).cpu().item())
                    if self.accumulated_rel_l1_distance_drop_text < self.teacache_thresh:
                        should_calc_drop_text = False
                    else:
                        should_calc_drop_text = True
                        self.accumulated_rel_l1_distance_drop_text = 0
                self.previous_e0_drop_text = modulated_inp.clone()
            else: # uncond
                if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_uncond = True
                    self.accumulated_rel_l1_distance_uncond = 0
                else:
                    rescale_func = np.poly1d(self.coefficients)
                    self.accumulated_rel_l1_distance_uncond += rescale_func(((modulated_inp-self.previous_e0_uncond).abs().mean() / self.previous_e0_uncond.abs().mean()).cpu().item())
                    if self.accumulated_rel_l1_distance_uncond < self.teacache_thresh:
                        should_calc_uncond = False
                    else:
                        should_calc_uncond = True
                        self.accumulated_rel_l1_distance_uncond = 0
                self.previous_e0_uncond = modulated_inp.clone()

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            audio_embedding=audio_embedding,
            ref_target_masks=token_ref_target_masks,
            human_num=human_num,
            )
        if self.enable_teacache:
            if self.cnt%3==0:
                if not should_calc_cond:
                    x +=  self.previous_residual_cond
                else:
                    ori_x = x.clone()
                    for block in self.blocks:
                        x = block(x, **kwargs)
                    self.previous_residual_cond = x - ori_x
            elif self.cnt%3==1:
                if not should_calc_drop_text:
                    x +=  self.previous_residual_drop_text
                else:
                    ori_x = x.clone()
                    for block in self.blocks:
                        x = block(x, **kwargs)
                    self.previous_residual_drop_text = x - ori_x
            else:
                if not should_calc_uncond:
                    x +=  self.previous_residual_uncond
                else:
                    ori_x = x.clone()
                    for block in self.blocks:
                        x = block(x, **kwargs)
                    self.previous_residual_uncond = x - ori_x
        else:
            for block in self.blocks:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        if self.enable_teacache:
            self.cnt += 1
            if self.cnt >= self.num_steps:
                self.cnt = 0

        return torch.stack(x).float()


    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)