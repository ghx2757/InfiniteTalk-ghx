# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
from inspect import ArgSpec
import logging
import json
import math
import importlib
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
from PIL import Image

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from diffusers.models.modeling_utils import no_init_weights, ContextManagers
import accelerate

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.multitalk_model import WanModel, WanLayerNorm, WanRMSNorm
from .modules.t5 import T5EncoderModel, T5LayerNorm, T5RelativeEmbedding
from .modules.vae import WanVAE, CausalConv3d, RMS_norm, Upsample
from .utils.multitalk_utils import MomentumBuffer, adaptive_projected_guidance, match_and_blend_colors, save_video_ffmpeg_noaudio
from src.vram_management import AutoWrappedQLinear, AutoWrappedLinear, AutoWrappedModule, enable_vram_management
from wan.utils.utils import convert_video_to_h264, extract_specific_frames, get_video_codec
from wan.wan_lora import WanLoraWrapper

from safetensors.torch import load_file
from optimum.quanto import quantize, freeze, qint8,requantize
import optimum.quanto.nn.qlinear as qlinear

def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def to_param_dtype_fp32only(model, param_dtype):
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param.dtype == torch.float32 and param.__class__.__name__ not in ['WeightQBytesTensor']:
                param.data = param.data.to(param_dtype)
        for name, buf in module.named_buffers(recurse=False):
            if buf.dtype == torch.float32 and buf.__class__.__name__ not in ['WeightQBytesTensor']:
                module._buffers[name] = buf.to(param_dtype)
                
def resize_and_centercrop(cond_image, target_size):
        """
        Resize image or tensor to the target size without padding.
        """

        # Get the original size
        if isinstance(cond_image, torch.Tensor):
            _, orig_h, orig_w = cond_image.shape
        else:
            orig_h, orig_w = cond_image.height, cond_image.width

        target_h, target_w = target_size
        
        # Calculate the scaling factor for resizing
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w
        
        # Compute the final size
        scale = max(scale_h, scale_w)
        final_h = math.ceil(scale * orig_h)
        final_w = math.ceil(scale * orig_w)
        
        # Resize
        if isinstance(cond_image, torch.Tensor):
            if len(cond_image.shape) == 3:
                cond_image = cond_image[None]
            resized_tensor = nn.functional.interpolate(cond_image, size=(final_h, final_w), mode='nearest').contiguous() 
            # crop
            cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size) 
            cropped_tensor = cropped_tensor.squeeze(0)
        else:
            resized_image = cond_image.resize((final_w, final_h), resample=Image.BILINEAR)
            resized_image = np.array(resized_image)
            # tensor and crop
            resized_tensor = torch.from_numpy(resized_image)[None, ...].permute(0, 3, 1, 2).contiguous()
            cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size)
            cropped_tensor = cropped_tensor[:, :, None, :, :] 

        return cropped_tensor


def timestep_transform(
    t,
    shift=5.0,
    num_timesteps=1000,
):
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t


'''
InfiniteTalkPipeline.__init__()
│
├─ 1. 基础配置（设备、rank、dtype等）
│
├─ 2. 文本编码器（T5）
│   └─ 支持FSDP、量化、CPU模式
│
├─ 3. VAE（视频编解码器）
│
├─ 4. CLIP（视觉-语言模型）
│
├─ 5. DiT主模型
│   ├─ 量化模式：从quant_dir加载
│   ├─ 分片模式：合并7个safetensors + infinitetalk权重
│   └─ checkpoint模式：从dit_path加载
│
├─ 6. 模型后处理
│   ├─ eval() + requires_grad_(False)
│   ├─ 数据类型转换
│   └─ 可选：加载LoRA权重
│
├─ 7. 分布式策略
│   ├─ USP：替换forward方法实现序列并行
│   └─ FSDP：模型分片
│
└─ 8. 最终配置（采样参数、显存管理标志）
'''
class InfiniteTalkPipeline:

    def __init__(
        self,
        config,
        checkpoint_dir,
        quant_dir=None,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
        num_timesteps=1000,
        use_timestep_transform=True,
        lora_dir=None,
        lora_scales=None,
        quant = None,
        dit_path = None,
        infinitetalk_dir=None,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            quant (`str`, *optional*, defaults to None):
                Quantization type, must be 'int8' or 'fp8'.
        """
        if quant is not None and quant not in ("int8", "fp8"):
            raise ValueError("quant must be 'int8', 'fp8', or None(default fp32 model)")
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)

        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
            quant=quant,
            quant_dir=os.path.dirname(quant_dir) if quant_dir is not None else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {checkpoint_dir}")

        if quant is not None:
            logging.info(f"Loading Quantized MultiTalk from {quant_dir}")
            with torch.device('meta'):
                wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
                self.model = WanModel(weight_init=False,**wan_config)
                torch_gc()
            model_state_dict = load_file(quant_dir)
            map_json_path = os.path.join(quant_dir.replace('safetensors', 'json'))
            self.model.init_freqs()
            with open(map_json_path, "r") as f:
                quantization_map = json.load(f)
            requantize(self.model, model_state_dict, quantization_map, device='cpu')
        else:
            if dit_path is None:
                init_contexts = [no_init_weights()]
                init_contexts.append(accelerate.init_empty_weights())
                wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
                self.model = WanModel(weight_init=False,**wan_config).to(dtype=self.param_dtype)
                weight_files = [f"{checkpoint_dir}/diffusion_pytorch_model-00001-of-00007.safetensors", 
                                f"{checkpoint_dir}/diffusion_pytorch_model-00002-of-00007.safetensors", 
                                f"{checkpoint_dir}/diffusion_pytorch_model-00003-of-00007.safetensors", 
                                f"{checkpoint_dir}/diffusion_pytorch_model-00004-of-00007.safetensors",
                                f"{checkpoint_dir}/diffusion_pytorch_model-00005-of-00007.safetensors", 
                                f"{checkpoint_dir}/diffusion_pytorch_model-00006-of-00007.safetensors", 
                                f"{checkpoint_dir}/diffusion_pytorch_model-00007-of-00007.safetensors",
                                f"{infinitetalk_dir}"]
                merged_state_dict = {}
                for weight_file in weight_files:
                    sd = load_file(weight_file)
                    merged_state_dict.update(sd)
                self.model.load_state_dict(merged_state_dict)
                
            else:
                init_contexts = [no_init_weights()]
                init_contexts.append(accelerate.init_empty_weights())
                with ContextManagers(init_contexts):
                    wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
                    self.model = WanModel(weight_init=False,**wan_config)
                checkpoint_weights = torch.load(dit_path, map_location='cpu')
                self.model.load_state_dict(checkpoint_weights['state_dict'])
                logging.info(f"loading infinitetalk weights {checkpoint_dir}")
            
        self.model.eval().requires_grad_(False)
        
        to_param_dtype_fp32only(self.model, self.param_dtype)
        if lora_dir is not None and quant is None :
            lora_wrapper = WanLoraWrapper(self.model)
            for lora_path, lora_scale in zip(lora_dir, lora_scales):
                lora_name = lora_wrapper.load_lora(lora_path)
                lora_wrapper.apply_lora(lora_name, lora_scale, param_dtype=self.param_dtype, device=self.device)


    

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_dit_forward_multitalk,
                usp_attn_forward_multitalk,
                usp_crossattn_multi_forward_multitalk
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward_multitalk, block.self_attn)
                block.audio_cross_attn.forward = types.MethodType(
                    usp_crossattn_multi_forward_multitalk, block.audio_cross_attn)
            self.model.forward = types.MethodType(usp_dit_forward_multitalk, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)
        
        self.sample_neg_prompt = config.sample_neg_prompt
        self.num_timesteps = num_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.cpu_offload = False
        self.model_names = ["model"]
        self.vram_management = False

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timesteps = timesteps.float() / self.num_timesteps
        timesteps = timesteps.view(timesteps.shape + (1,) * (len(noise.shape)-1))

        return (1 - timesteps) * original_samples + timesteps * noise

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.model.parameters())).dtype
        enable_vram_management(
            self.model,
            module_map={
                qlinear.QLinear: AutoWrappedQLinear,
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                WanLayerNorm: AutoWrappedModule,
                WanRMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.param_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.param_dtype,
                computation_device=self.device,
            ),
        )
        self.enable_cpu_offload()

    def enable_cpu_offload(self):
        self.cpu_offload = True
    
    def load_models_to_device(self, loadmodel_names=[]):
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            return
        # offload the unneeded models to cpu
        for model_name in self.model_names:
            if model_name not in loadmodel_names:
                model = getattr(self, model_name)

                if not isinstance(model, nn.Module):
                    model = model.model

                if model is not None:
                    if (
                        hasattr(model, "vram_management_enabled")
                        and model.vram_management_enabled
                    ):
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if not isinstance(model, nn.Module):
                model = model.model
            if model is not None:
                if (
                    hasattr(model, "vram_management_enabled")
                    and model.vram_management_enabled
                ):
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()
                else:
                    model.to(self.device)
        # fresh the cuda cache
        torch.cuda.empty_cache()

   
    def generate_infinitetalk(self,
                 input_data,
                 size_buckget='infinitetalk-480',
                 motion_frame=25,
                 frame_num=81,
                 shift=5.0,
                 sampling_steps=40,
                 text_guide_scale=5.0,
                 audio_guide_scale=4.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 max_frames_num=1000,
                 face_scale=0.05,
                 progress=True,
                 color_correction_strength=0.0,
                 save_file=None,
                 extra_args=None):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
        """

        # init teacache
        if extra_args.use_teacache:
            self.model.teacache_init(
                sample_steps=sampling_steps,
                teacache_thresh=extra_args.teacache_thresh,
                model_scale=extra_args.size,
            )
        else:
            self.model.disable_teacache()

        input_prompt = input_data['prompt']
        cond_file_path = input_data['cond_video']
        codec = get_video_codec(cond_file_path)
        if codec == 'av1': # 视频编码检测与转换
            output_video_path = 'tmp/' + '_input_h264.mp4'
            print(f"Converting {cond_file_path} from AV1 to H.264...")
            convert_video_to_h264(cond_file_path, output_video_path)
            print(f"Conversion complete! Saved as {output_video_path}")
            cond_file_path = output_video_path
        else:
            print("No conversion needed.")
        cond_image = extract_specific_frames(cond_file_path, 0) # 【提取第一帧作为条件图像】
        # cond_image = Image.fromarray(cond_image)
        
        
        # decide a proper size
        bucket_config_module = importlib.import_module("wan.utils.multitalk_utils")
        if size_buckget == 'infinitetalk-480':
            bucket_config = getattr(bucket_config_module, 'ASPECT_RATIO_627')
        elif size_buckget == 'infinitetalk-720':
            bucket_config = getattr(bucket_config_module, 'ASPECT_RATIO_960')
        elif size_buckget == 'infinitetalk-512':    
            bucket_config = getattr(bucket_config_module, 'ASPECT_RATIO_512')

        # 计算最接近的bucket → (target_h, target_w)，对参考图进行处理
        src_h, src_w = cond_image.height, cond_image.width
        ratio = src_h / src_w
        closest_bucket = sorted(list(bucket_config.keys()), key=lambda x: abs(float(x)-ratio))[0]
        target_h, target_w = bucket_config[closest_bucket][0]
        cond_image = resize_and_centercrop(cond_image, (target_h, target_w))
        cond_image = cond_image / 255
        cond_image = (cond_image - 0.5) * 2 # normalization
        cond_image = cond_image.to(self.device)  # 1 C 1 H W

        # Store the original image for color reference if strength > 0
        original_color_reference = None
        if color_correction_strength > 0.0:
            original_color_reference = cond_image.clone() #【保存当前参考图用于颜色校正】


        # read audio embeddings
        audio_embedding_path_1 = input_data['cond_audio']['person1']
        if len(input_data['cond_audio']) == 1:
            HUMAN_NUMBER = 1
            audio_embedding_path_2 = None
        else:
            HUMAN_NUMBER = 2
            audio_embedding_path_2 = input_data['cond_audio']['person2']

        
        full_audio_embs = [] # 【驱动音频embedding】        
        audio_embedding_paths = [audio_embedding_path_1, audio_embedding_path_2]
        for human_idx in range(HUMAN_NUMBER):   
            audio_embedding_path = audio_embedding_paths[human_idx]
            if not os.path.exists(audio_embedding_path):
                continue
            full_audio_emb = torch.load(audio_embedding_path)
            if torch.isnan(full_audio_emb).any():
                continue
            if full_audio_emb.shape[0] <= frame_num:
                continue
            full_audio_embs.append(full_audio_emb) 
        
        assert len(full_audio_embs) == HUMAN_NUMBER, f"Aduio file not exists or length not satisfies frame nums."

        # preprocess text embedding
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context, context_null = self.text_encoder([input_prompt, n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu')) # 【正向文本编码】
            context_null = self.text_encoder([n_prompt], torch.device('cpu')) # 【负向文本编码】
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        torch_gc()
        # prepare params for video generation
        indices = (torch.arange(2 * 2 + 1) - 2) * 1  # 【indices = [-2,-1,0,1,2]】
        clip_length = frame_num # 81
        is_first_clip = True
        arrive_last_frame = False # 到达最后帧标识位
        cur_motion_frames_num = 1
        audio_start_idx = 0
        audio_end_idx = audio_start_idx + clip_length
        gen_video_list = []

        # Initialize variables for segmented video saving
        segment_counter = 0 # 视频片段计数器
        saved_segments = [] # 保存视频片段的路径列表
        save_dir = None # 视频保存目录
        if save_file is not None:
            save_dir = save_file
            os.makedirs(save_dir, exist_ok=True)
        torch_gc()

        # set random seed and init noise
        seed = seed if seed >= 0 else random.randint(0, 99999999)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        # start video generat-ion iteratively
        while True: # 针对clip_length的视频切片循环生成完整视频
            audio_embs = []
            # split audio with window size
            for human_idx in range(HUMAN_NUMBER):   
                center_indices = torch.arange(
                    audio_start_idx,
                    audio_end_idx,
                    1,
                ).unsqueeze(
                    1
                ) + indices.unsqueeze(0)
                center_indices = torch.clamp(center_indices, min=0, max=full_audio_embs[human_idx].shape[0]-1)
                audio_emb = full_audio_embs[human_idx][center_indices][None,...].to(self.device)
                audio_embs.append(audio_emb)
            audio_embs = torch.concat(audio_embs, dim=0).to(self.param_dtype) # 【[N,T,5,D]】
            torch_gc()

            h, w = cond_image.shape[-2], cond_image.shape[-1]
            lat_h, lat_w = h // self.vae_stride[1], w // self.vae_stride[2]
            max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
                self.patch_size[1] * self.patch_size[2])
            max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size



            noise = torch.randn( # 【噪声】
                16, (frame_num - 1) // 4 + 1,
                lat_h,
                lat_w,
                dtype=torch.float32,
                device=self.device) 

            # get mask
            
            msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device) # 【mask】
            msk[:, 1:] = 0
            msk = torch.concat([
                torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
            ],
                            dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2).to(self.param_dtype) # B 4 T H W

            with torch.no_grad():
                # get clip embedding
                self.clip.model.to(self.device)
                clip_context = self.clip.visual(cond_image[:, :, -1:, :, :]).to(self.param_dtype) # 【CLIP视觉特征】
                if offload_model:
                    self.clip.model.cpu()
                torch_gc()

                # zero padding and vae encode
                video_frames = torch.zeros(1, cond_image.shape[1], frame_num-cond_image.shape[2], target_h, target_w).to(self.device)
                padding_frames_pixels_values = torch.concat([cond_image, video_frames], dim=2)
                y = self.vae.encode(padding_frames_pixels_values) # 【Y-Z2】
                y = torch.stack(y).to(self.param_dtype) # B C T H W
                cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num-1) // 4)

                if is_first_clip:
                    latent_motion_frames = self.vae.encode(cond_image)[0] # 【潜在运动编码初始化】
                else:
                    latent_motion_frames = self.vae.encode(cond_frame)[0]

                y = torch.concat([msk, y], dim=1) # B 4+C T H W
                torch_gc()
            

            # construct human mask
            human_masks = []
            if HUMAN_NUMBER==1:
                background_mask = torch.ones([src_h, src_w])
                human_mask1 = torch.ones([src_h, src_w])
                human_mask2 = torch.ones([src_h, src_w])
                human_masks = [human_mask1, human_mask2, background_mask]
            elif HUMAN_NUMBER==2:
                if 'bbox' in input_data:
                    assert len(input_data['bbox']) == len(input_data['cond_audio']), f"The number of target bbox should be the same with cond_audio"
                    background_mask = torch.zeros([src_h, src_w])
                    for _, person_bbox in input_data['bbox'].items():
                        x_min, y_min, x_max, y_max = person_bbox
                        human_mask = torch.zeros([src_h, src_w])
                        human_mask[int(x_min):int(x_max), int(y_min):int(y_max)] = 1
                        background_mask += human_mask
                        human_masks.append(human_mask)
                else:
                    x_min, x_max = int(src_h * face_scale), int(src_h * (1 - face_scale))
                    background_mask = torch.zeros([src_h, src_w])
                    background_mask = torch.zeros([src_h, src_w])
                    human_mask1 = torch.zeros([src_h, src_w])
                    human_mask2 = torch.zeros([src_h, src_w])
                    lefty_min, lefty_max = int((src_w//2) * face_scale), int((src_w//2) * (1 - face_scale))
                    righty_min, righty_max = int((src_w//2) * face_scale + (src_w//2)), int((src_w//2) * (1 - face_scale) + (src_w//2))
                    human_mask1[x_min:x_max, lefty_min:lefty_max] = 1
                    human_mask2[x_min:x_max, righty_min:righty_max] = 1
                    background_mask += human_mask1
                    background_mask += human_mask2
                    human_masks = [human_mask1, human_mask2]
                background_mask = torch.where(background_mask > 0, torch.tensor(0), torch.tensor(1))
                human_masks.append(background_mask)

            ref_target_masks = torch.stack(human_masks, dim=0).to(self.device)
            # resize and centercrop for ref_target_masks 
            ref_target_masks = resize_and_centercrop(ref_target_masks, (target_h, target_w))

            _, _, _,lat_h, lat_w = y.shape
            ref_target_masks = F.interpolate(ref_target_masks.unsqueeze(0), size=(lat_h, lat_w), mode='nearest').squeeze() 
            ref_target_masks = (ref_target_masks > 0) 
            ref_target_masks = ref_target_masks.float().to(self.device)

            torch_gc()

            @contextmanager
            def noop_no_sync():
                yield

            no_sync = getattr(self.model, 'no_sync', noop_no_sync)

            # evaluation mode
            with torch.no_grad(), no_sync():
                
                # prepare timesteps
                timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
                timesteps.append(0.)
                timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
                if self.use_timestep_transform:
                    timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps]
                
                # sample videos
                latent = noise

                # prepare condition and uncondition configs
                arg_c = { # 完整条件
                    'context': [context],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': audio_embs,
                    'ref_target_masks': ref_target_masks
                }


                arg_null_text = { # 去文本
                    'context': [context_null],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': audio_embs,
                    'ref_target_masks': ref_target_masks
                }

                arg_null_audio = { # 去音频
                    'context': [context],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': torch.zeros_like(audio_embs)[-1:],
                    'ref_target_masks': ref_target_masks
                }


                arg_null = { # 全无条件
                    'context': [context_null],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': torch.zeros_like(audio_embs)[-1:],
                    'ref_target_masks': ref_target_masks
                }

                torch_gc()
                if not self.vram_management:
                    self.model.to(self.device)
                else:
                    self.load_models_to_device(["model"])
                
                # injecting motion frames
                if not is_first_clip:
                    latent_motion_frames = latent_motion_frames.to(latent.dtype).to(self.device)
                    motion_add_noise = torch.randn_like(latent_motion_frames).contiguous()
                    add_latent = self.add_noise(latent_motion_frames, motion_add_noise, timesteps[0])
                    _, T_m, _, _ = add_latent.shape
                    latent[:, :T_m] = add_latent

                # infer with APG
                # refer https://arxiv.org/abs/2410.02416   
                if extra_args.use_apg:  
                    text_momentumbuffer  = MomentumBuffer(extra_args.apg_momentum) 
                    audio_momentumbuffer = MomentumBuffer(extra_args.apg_momentum) 


                progress_wrap = partial(tqdm, total=len(timesteps)-1) if progress else (lambda x: x)
                for i in progress_wrap(range(len(timesteps)-1)): # 遍历采样时间步
                    timestep = timesteps[i]
                    latent[:, :cur_motion_frames_latent_num] = latent_motion_frames # 逻辑重复了？
                    latent_model_input = [latent.to(self.device)]

                    # inference with CFG strategy
                    noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]  # 【完整条件下的噪声预测】
                    torch_gc()

                    if math.isclose(text_guide_scale, 1.0):
                        noise_pred_drop_audio = self.model(
                            latent_model_input, t=timestep, **arg_null_audio)[0]  # 【去音频条件下的噪声预测】
                        torch_gc()
                    else:
                        noise_pred_drop_text = self.model(
                            latent_model_input, t=timestep, **arg_null_text)[0]  # 【去文本条件下的噪声预测】
                        torch_gc()
                        noise_pred_uncond = self.model(
                            latent_model_input, t=timestep, **arg_null)[0]   # 【全无条件下的噪声预测】
                        torch_gc()

                    if extra_args.use_apg:
                        # correct update direction
                        if math.isclose(text_guide_scale, 1.0):
                            diff_uncond_audio  = noise_pred_cond - noise_pred_drop_audio
                            noise_pred = noise_pred_cond + (audio_guide_scale - 1)* adaptive_projected_guidance(diff_uncond_audio, 
                                                                                            noise_pred_cond, 
                                                                                            momentum_buffer=audio_momentumbuffer, 
                                                                                            norm_threshold=extra_args.apg_norm_threshold)
                        else:
                            diff_uncond_text  = noise_pred_cond - noise_pred_drop_text
                            diff_uncond_audio = noise_pred_drop_text - noise_pred_uncond
                            noise_pred = noise_pred_cond + (text_guide_scale - 1) * adaptive_projected_guidance(diff_uncond_text, 
                                                                                                                noise_pred_cond, 
                                                                                                                momentum_buffer=text_momentumbuffer, 
                                                                                                                norm_threshold=extra_args.apg_norm_threshold) \
                                + (audio_guide_scale - 1) * adaptive_projected_guidance(diff_uncond_audio, 
                                                                                            noise_pred_cond, 
                                                                                            momentum_buffer=audio_momentumbuffer, 
                                                                                            norm_threshold=extra_args.apg_norm_threshold)
                    else:
                        # vanilla CFG strategy
                        if math.isclose(text_guide_scale, 1.0):
                            noise_pred = noise_pred_drop_audio + audio_guide_scale* (noise_pred_cond - noise_pred_drop_audio)  
                        else:
                            noise_pred = noise_pred_uncond + text_guide_scale * (
                                noise_pred_cond - noise_pred_drop_text) + \
                                audio_guide_scale * (noise_pred_drop_text - noise_pred_uncond)  
                    noise_pred = -noise_pred  

                    # update latent
                    dt = timesteps[i] - timesteps[i + 1]
                    dt = dt / self.num_timesteps
                    latent = latent + noise_pred * dt[:, None, None, None]

                    # injecting motion frames
                    if not is_first_clip:
                        latent_motion_frames = latent_motion_frames.to(latent.dtype).to(self.device)
                        motion_add_noise = torch.randn_like(latent_motion_frames).contiguous()
                        add_latent = self.add_noise(latent_motion_frames, motion_add_noise, timesteps[i+1])
                        _, T_m, _, _ = add_latent.shape
                        latent[:, :T_m] = add_latent

                    latent[:, :cur_motion_frames_latent_num] = latent_motion_frames # 逻辑重复了？
                    x0 = [latent.to(self.device)] 
                    del latent_model_input, timestep
                
                if offload_model: 
                    if not self.vram_management:
                        self.model.cpu()
                torch_gc()

                videos = self.vae.decode(x0)
            
            # cache generated samples
            videos = torch.stack(videos).cpu() # B C T H W
            # >>> START OF COLOR CORRECTION STEP <<<
            if color_correction_strength > 0.0 and original_color_reference is not None:
                videos = match_and_blend_colors(videos, original_color_reference, color_correction_strength)
            # >>> END OF COLOR CORRECTION STEP <<<

            if is_first_clip:
                gen_video_list.append(videos)
            else:
                gen_video_list.append(videos[:, :, cur_motion_frames_num:])

            # Segmented video saving logic --by ghx
            if save_dir is None:
                raise ValueError("===> save_file path must be provided for saving segmented videos.")            
            if is_first_clip:
                
                gen_video = torch.cat(gen_video_list, dim=2)
                gen_video = gen_video.to(torch.float32)  

                segment_path = os.path.join(save_dir, f"{segment_counter:04d}")
                save_video_ffmpeg_noaudio(gen_video[0], segment_path, high_quality_save=True)
                saved_segments.append(segment_path)

                gen_video_list.clear()

            elif segment_counter % 20 == 19:  # Save every 20 segments (0-based: 19, 39, 59...)
            # else:
                gen_video = torch.cat(gen_video_list, dim=2)
                gen_video = gen_video.to(torch.float32)  

                segment_path = os.path.join(save_dir, f"{segment_counter:04d}")
                save_video_ffmpeg_noaudio(gen_video[0], segment_path, high_quality_save=True)
                saved_segments.append(segment_path)

                gen_video_list.clear()

            segment_counter += 1

            # decide whether is done
            if arrive_last_frame: break

            # update next condition frames
            is_first_clip = False
            cur_motion_frames_num = motion_frame

            cond_frame = videos[:, :, -cur_motion_frames_num:].to(torch.float32).to(self.device) # 【更新运动帧】
            audio_start_idx += (frame_num - cur_motion_frames_num)
            audio_end_idx = audio_start_idx + clip_length

            cond_image = extract_specific_frames(cond_file_path, audio_start_idx)
            # cond_image = Image.fromarray(cond_image)
            cond_image = resize_and_centercrop(cond_image, (target_h, target_w))
            cond_image = cond_image / 255
            cond_image = (cond_image - 0.5) * 2 # normalization
            cond_image = cond_image.to(self.device)  # 1 C 1 H W

            # Repeat audio emb
            if audio_end_idx >= min(max_frames_num, len(full_audio_embs[0])):
                arrive_last_frame = True
                miss_lengths = []
                source_frames = []
                for human_inx in range(HUMAN_NUMBER):
                    source_frame = len(full_audio_embs[human_inx])
                    source_frames.append(source_frame)
                    if audio_end_idx >= len(full_audio_embs[human_inx]):
                        miss_length   = audio_end_idx - len(full_audio_embs[human_inx]) + 3 
                        add_audio_emb = torch.flip(full_audio_embs[human_inx][-1*miss_length:], dims=[0])
                        full_audio_embs[human_inx] = torch.cat([full_audio_embs[human_inx], add_audio_emb], dim=0)
                        miss_lengths.append(miss_length)
                    else:
                        miss_lengths.append(0)

            
            if max_frames_num <= frame_num: break
            
            torch_gc()
            if offload_model:    
                torch.cuda.synchronize()
            if dist.is_initialized():
                dist.barrier()
        
        # gen_video_samples = torch.cat(gen_video_list, dim=2)[:, :, :int(max_frames_num)] 
        # gen_video_samples = gen_video_samples.to(torch.float32)
        # if max_frames_num > frame_num and sum(miss_lengths) > 0:
        #     # split video frames
        #     # gen_video_samples = gen_video_samples[:, :, :-1*miss_lengths[0]]
        #     gen_video_samples = gen_video_samples[:, :, :full_audio_emb.shape[0]]
        
        if dist.is_initialized():
            dist.barrier()

        del noise, latent
        torch_gc()

        # Save remaining segments if any
        if save_dir is not None and len(gen_video_list) > 0:
            gen_video = torch.cat(gen_video_list, dim=2)
            gen_video = gen_video.to(torch.float32)
            segment_path = os.path.join(save_dir, f"{segment_counter:04d}")
            save_video_ffmpeg_noaudio(gen_video[0], segment_path, high_quality_save=True)
            saved_segments.append(segment_path)
            gen_video_list.clear()

        # Save segment list to txt file -- by ghx
        if save_dir is not None and self.rank == 0:
            txt_path = os.path.join(save_dir, "clips.txt")
            with open(txt_path, 'w') as f:
                for segment_path in saved_segments:
                    # Check if the file exists with .mp4 extension
                    segment_path_with_ext = segment_path + '.mp4'
                    if os.path.exists(segment_path_with_ext):
                        relative_path = os.path.relpath(segment_path_with_ext, save_dir)
                    else:
                        # Try without extension (if save_video_ffmpeg_noaudio doesn't add .mp4)
                        relative_path = os.path.relpath(segment_path, save_dir)
                        if not os.path.exists(segment_path):
                            # Try to find the actual file
                            for ext in ['.mp4', '.avi', '.mov']:
                                test_path = segment_path + ext
                                if os.path.exists(test_path):
                                    relative_path = os.path.relpath(test_path, save_dir)
                                    break
                            else:
                                logging.warning(f"Segment file not found: {segment_path}")
                                continue
                    f.write(f"file '{relative_path}'\n")

        # return gen_video_samples[0] if self.rank == 0 else None
        return None
    

   
# ═══════════════════════════════════════════════════════════════
# generate_infinitetalk() - 基于扩散模型的说话人视频生成
# ═══════════════════════════════════════════════════════════════

# 输入: input_data {prompt, cond_video, cond_audio{person1, person2?}, bbox?}
#       + 生成配置参数
# │
# ├─【阶段1: 初始化与资源准备】(L439-510)
# │  │
# │  ├─ extra_args.use_teacache?
# │  │  ├─ 是 → teacache_init() 加速采样
# │  │  └─ 否 → disable_teacache()
# │  │
# │  ├─ 视频编码检测
# │  │  ├─ codec == 'av1'? → convert_to_h264() 转码
# │  │  └─ 否 → 保持原样
# │  │
# │  ├─ 提取参考帧: extract_specific_frames(cond_video, frame=0)
# │  │
# │  ├─ 图像预处理
# │  │  ├─ size_buckget 选择?
# │  │  │  ├─ 'infinitetalk-480' → ASPECT_RATIO_627
# │  │  │  └─ 'infinitetalk-720' → ASPECT_RATIO_960
# │  │  ├─ 计算最接近的bucket → (target_h, target_w)
# │  │  ├─ resize_and_centercrop()
# │  │  ├─ 归一化: /255 → [-1,1]
# │  │  └─ color_correction_strength > 0?
# │  │     ├─ 是 → 保存 original_color_reference
# │  │     └─ 否 → 跳过
# │  │
# │  └─ 音频嵌入加载
# │     ├─ len(cond_audio) == 1? → HUMAN_NUMBER=1, audio_path_2=None
# │     └─ len(cond_audio) == 2? → HUMAN_NUMBER=2, 加载两个音频
# │        └─ 验证: 文件存在 & 无NaN & 长度>=frame_num
# │           └─ full_audio_embs = [audio1, audio2?]
# │
# ├─【阶段2: 文本编码】(L511-524)
# │  │
# │  ├─ n_prompt为空? → 使用 self.sample_neg_prompt
# │  │
# │  ├─ self.t5_cpu?
# │  │  ├─ 否 (GPU模式)
# │  │  │  ├─ text_encoder → GPU
# │  │  │  ├─ encode([prompt, n_prompt])
# │  │  │  └─ offload_model? → text_encoder → CPU
# │  │  │
# │  │  └─ 是 (CPU模式)
# │  │     ├─ encode on CPU
# │  │     └─ context/context_null → GPU
# │  │
# │  └─ 输出: context, context_null
# │
# ├─【阶段3: 生成参数初始化】(L526-544)
# │  │
# │  ├─ 音频窗口参数
# │  │  ├─ indices = [-2,-1,0,1,2] * 1
# │  │  ├─ clip_length = frame_num
# │  │  ├─ audio_start_idx = 0
# │  │  └─ audio_end_idx = audio_start_idx + clip_length
# │  │
# │  ├─ 状态标志
# │  │  ├─ is_first_clip = True
# │  │  ├─ arrive_last_frame = False
# │  │  ├─ cur_motion_frames_num = 1
# │  │  └─ gen_video_list = []
# │  │
# │  └─ 随机种子设置
# │     ├─ seed >= 0? → 使用指定seed
# │     └─ seed == -1? → random.randint(0, 99999999)
# │        └─ 设置: torch, cuda, numpy, random, cudnn.deterministic
# │
# ├─【阶段4: 循环生成视频】while True (L546-865)
# │  │
# │  ├─ 4.1 音频窗口切分 (L547-561)
# │  │   │
# │  │   └─ for human_idx in range(HUMAN_NUMBER):
# │  │      ├─ 生成 center_indices (当前窗口 + 左右各2帧上下文)
# │  │      ├─ 裁剪到有效范围: clamp(0, audio_len-1)
# │  │      └─ audio_emb = full_audio_emb[center_indices]
# │  │         └─ audio_embs = concat([audio1, audio2?]) → [N,T,5,D]
# │  │
# │  ├─ 4.2 潜空间准备 (L563-610)
# │  │   │
# │  │   ├─ 计算潜空间尺寸
# │  │   │  ├─ lat_h = h // vae_stride[1]
# │  │   │  ├─ lat_w = w // vae_stride[2]
# │  │   │  └─ max_seq_len (考虑patch_size和sp_size)
# │  │   │
# │  │   ├─ 生成初始噪声: randn(16, (frame_num-1)//4+1, lat_h, lat_w)
# │  │   │
# │  │   ├─ 构建mask (第一帧为条件帧)
# │  │   │  ├─ msk = ones(1, frame_num, lat_h, lat_w)
# │  │   │  ├─ msk[:, 1:] = 0  (只保留第一帧)
# │  │   │  └─ 重排维度 → [B,4,T,H,W]
# │  │   │
# │  │   └─ with torch.no_grad():
# │  │      ├─ CLIP编码: clip_context = clip.visual(cond_image)
# │  │      ├─ VAE编码: 
# │  │      │  ├─ zero_padding → [cond_image, zeros...]
# │  │      │  └─ y = vae.encode(padding_frames)
# │  │      │
# │  │      ├─ is_first_clip?
# │  │      │  ├─ 是 → latent_motion_frames = vae.encode(cond_image)
# │  │      │  └─ 否 → latent_motion_frames = vae.encode(cond_frame)
# │  │      │
# │  │      └─ y = concat([msk, y], dim=1) → [B,4+C,T,H,W]
# │  │
# │  ├─ 4.3 人物Mask构建 (L612-653)
# │  │   │
# │  │   ├─ HUMAN_NUMBER == 1?
# │  │   │  └─ human_masks = [ones, ones, ones] (全1，无分割)
# │  │   │
# │  │   └─ HUMAN_NUMBER == 2?
# │  │      │
# │  │      ├─ 'bbox' in input_data?
# │  │      │  │
# │  │      │  ├─ 是 (精确模式)
# │  │      │  │  └─ for person_bbox in input_data['bbox']:
# │  │      │  │     ├─ human_mask[x_min:x_max, y_min:y_max] = 1
# │  │      │  │     └─ human_masks.append(human_mask)
# │  │      │  │
# │  │      │  └─ 否 (自动分割模式)
# │  │      │     ├─ 左半区域: human_mask1[x_min:x_max, lefty_min:lefty_max]
# │  │      │     └─ 右半区域: human_mask2[x_min:x_max, righty_min:righty_max]
# │  │      │
# │  │      └─ background_mask = ~(human_mask1 + human_mask2)
# │  │         └─ human_masks = [mask1, mask2, bg_mask]
# │  │            │
# │  │            ├─ resize_and_centercrop() → (target_h, target_w)
# │  │            ├─ interpolate() → (lat_h, lat_w)
# │  │            └─ ref_target_masks (float tensor)
# │  │
# │  ├─ 4.4 扩散去噪采样 (L656-801)
# │  │   │
# │  │   └─ with torch.no_grad(), no_sync():
# │  │      │
# │  │      ├─ 准备时间步
# │  │      │  ├─ timesteps = linspace(num_timesteps, 1, sampling_steps)
# │  │      │  ├─ append(0.)
# │  │      │  └─ use_timestep_transform? → timestep_transform(shift)
# │  │      │
# │  │      ├─ 准备条件配置 (4种)
# │  │      │  ├─ arg_c:          {context, clip, y, audio, masks} (完整条件)
# │  │      │  ├─ arg_null_text:  {context_null, ..., audio, ...} (去文本)
# │  │      │  ├─ arg_null_audio: {context, ..., zeros, ...}      (去音频)
# │  │      │  └─ arg_null:       {context_null, ..., zeros, ...} (完全无条件)
# │  │      │
# │  │      ├─ 模型加载
# │  │      │  ├─ vram_management? → load_models_to_device(["model"])
# │  │      │  └─ 否 → model.to(device)
# │  │      │
# │  │      ├─ 注入运动帧 (非首片段)
# │  │      │  └─ not is_first_clip?
# │  │      │     ├─ add_noise(latent_motion_frames, noise, t[0])
# │  │      │     └─ latent[:, :T_m] = add_latent
# │  │      │
# │  │      ├─ extra_args.use_apg? → 初始化MomentumBuffer
# │  │      │  ├─ text_momentumbuffer
# │  │      │  └─ audio_momentumbuffer
# │  │      │
# │  │      └─ for i in range(len(timesteps)-1):  【主循环】
# │  │         │
# │  │         ├─ 注入运动帧约束
# │  │         │  └─ latent[:, :cur_motion_frames_latent_num] = latent_motion_frames
# │  │         │
# │  │         ├─ 模型预测 (CFG策略)
# │  │         │  ├─ noise_pred_cond = model(latent, t, **arg_c)
# │  │         │  │
# │  │         │  ├─ text_guide_scale ≈ 1.0? (纯音频模式)
# │  │         │  │  └─ noise_pred_drop_audio = model(latent, t, **arg_null_audio)
# │  │         │  │
# │  │         │  └─ text_guide_scale > 1.0? (文本+音频模式)
# │  │         │     ├─ noise_pred_drop_text = model(latent, t, **arg_null_text)
# │  │         │     └─ noise_pred_uncond = model(latent, t, **arg_null)
# │  │         │
# │  │         ├─ 计算最终噪声预测
# │  │         │  │
# │  │         │  ├─ extra_args.use_apg? (自适应投影引导)
# │  │         │  │  │
# │  │         │  │  ├─ text_guide_scale ≈ 1.0?
# │  │         │  │  │  └─ diff = cond - drop_audio
# │  │         │  │  │     └─ noise_pred = cond + (audio_scale-1) * APG(diff, cond, audio_momentum)
# │  │         │  │  │
# │  │         │  │  └─ text_guide_scale > 1.0?
# │  │         │  │     ├─ diff_text = cond - drop_text
# │  │         │  │     ├─ diff_audio = drop_text - uncond
# │  │         │  │     └─ noise_pred = cond 
# │  │         │  │        + (text_scale-1) * APG(diff_text, cond, text_momentum)
# │  │         │  │        + (audio_scale-1) * APG(diff_audio, cond, audio_momentum)
# │  │         │  │
# │  │         │  └─ 否 (经典CFG)
# │  │         │     │
# │  │         │     ├─ text_guide_scale ≈ 1.0?
# │  │         │     │  └─ noise_pred = drop_audio + audio_scale * (cond - drop_audio)
# │  │         │     │
# │  │         │     └─ text_guide_scale > 1.0?
# │  │         │        └─ noise_pred = uncond 
# │  │         │           + text_scale * (cond - drop_text)
# │  │         │           + audio_scale * (drop_text - uncond)
# │  │         │
# │  │         ├─ 更新潜变量
# │  │         │  ├─ dt = (timesteps[i] - timesteps[i+1]) / num_timesteps
# │  │         │  └─ latent = latent + noise_pred * dt
# │  │         │
# │  │         ├─ 重新注入运动帧 (非首片段)
# │  │         │  └─ not is_first_clip?
# │  │         │     ├─ add_noise(latent_motion_frames, noise, t[i+1])
# │  │         │     └─ latent[:, :T_m] = add_latent
# │  │         │
# │  │         └─ latent[:, :cur_motion_frames_latent_num] = latent_motion_frames
# │  │            └─ x0 = [latent]
# │  │
# │  ├─ 4.5 VAE解码与后处理 (L804-822)
# │  │   │
# │  │   ├─ offload_model? → model.cpu()
# │  │   │
# │  │   ├─ videos = vae.decode(x0) → [B,C,T,H,W]
# │  │   │
# │  │   ├─ color_correction_strength > 0?
# │  │   │  └─ videos = match_and_blend_colors(videos, original_ref, strength)
# │  │   │
# │  │   └─ 保存视频片段
# │  │      ├─ is_first_clip? → gen_video_list.append(videos)
# │  │      └─ 否 → gen_video_list.append(videos[:,:,cur_motion_frames_num:])
# │  │
# │  ├─ 4.6 终止判断 (L824)
# │  │   │
# │  │   └─ arrive_last_frame? → break 退出循环
# │  │
# │  └─ 4.7 准备下一轮迭代 (L826-860)
# │     │
# │     ├─ 更新状态标志
# │     │  ├─ is_first_clip = False
# │     │  └─ cur_motion_frames_num = motion_frame
# │     │
# │     ├─ 更新条件帧
# │     │  └─ cond_frame = videos[:,:,-cur_motion_frames_num:]
# │     │
# │     ├─ 更新音频窗口
# │     │  ├─ audio_start_idx += (frame_num - cur_motion_frames_num)
# │     │  └─ audio_end_idx = audio_start_idx + clip_length
# │     │
# │     ├─ 提取新参考帧
# │     │  ├─ extract_specific_frames(cond_video, audio_start_idx)
# │     │  ├─ resize_and_centercrop()
# │     │  └─ 归一化到[-1,1]
# │     │
# │     ├─ 检查是否到达末尾
# │     │  │
# │     │  └─ audio_end_idx >= min(max_frames_num, audio_len)?
# │     │     ├─ arrive_last_frame = True
# │     │     └─ for human_idx:
# │     │        └─ audio_end_idx >= len(audio_emb)?
# │     │           ├─ miss_length = audio_end_idx - len(audio_emb) + 3
# │     │           ├─ add_audio = flip(audio[-miss_length:])  (镜像填充)
# │     │           └─ full_audio_emb = cat([audio, add_audio])
# │     │
# │     └─ max_frames_num <= frame_num? → break
# │
# └─【阶段5: 最终合成与清理】(L866-879)
#    │
#    ├─ 拼接所有片段
#    │  └─ gen_video_samples = cat(gen_video_list, dim=2)[:,:,:max_frames_num]
#    │
#    ├─ 裁剪到实际音频长度
#    │  └─ max_frames_num > frame_num & sum(miss_lengths) > 0?
#    │     └─ gen_video_samples = gen_video_samples[:,:,:full_audio_emb.shape[0]]
#    │
#    ├─ 清理资源
#    │  ├─ del noise, latent
#    │  └─ torch_gc()
#    │
#    └─ 输出: gen_video_samples[0] [C,T,H,W] 或 None (非rank0进程)

# ═══════════════════════════════════════════════════════════════
# 核心技术特点:
# • 滑动窗口 + 运动帧注入 → 长视频连续生成
# • 双CFG (文本+音频) → 精细控制
# • APG加速 → 提升质量
# • 多人mask分离 → 双人对话
# • 音频镜像填充 → 处理边界
# ═══════════════════════════════════════════════════════════════