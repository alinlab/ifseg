# Modified from OFA

# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import math
import random
from typing import Any, Dict, List, Optional, Tuple
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    SinusoidalPositionalEmbedding,
    GradMultiply
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
from einops import rearrange, repeat

from .unify_transformer_layer import TransformerDecoderLayer
from .resnet import ResNet
from .frozen_bn import FrozenBatchNorm2d

import logging
logger = logging.getLogger(__name__)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

def resolve_str_true_false(x):
    x = x.lower()
    if x == 'true':
        return True
    elif x == 'false':
        return False
    else:
        raise ValueError(f"Unable to recognize string bool input: {x}")

def maybe_no_grad(no_grad=True):

    if no_grad:
        return torch.no_grad()
    else:
        return contextlib.ExitStack()  # dummy contextmanager

def make_token_bucket_position(bucket_size, max_position=DEFAULT_MAX_SOURCE_POSITIONS):
    context_pos = torch.arange(max_position, dtype=torch.long)[:, None]
    memory_pos = torch.arange(max_position, dtype=torch.long)[None, :]
    relative_pos = context_pos - memory_pos
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    
    abs_pos = torch.where((relative_pos<mid) & (relative_pos > -mid), mid-1, torch.abs(relative_pos))
    
    log_pos = torch.ceil(torch.log(abs_pos/mid)/math.log((max_position-1)/mid) * (mid-1)) + mid
    
    log_pos = log_pos.int()
    bucket_pos = torch.where(abs_pos.le(mid), relative_pos, log_pos*sign).long()
    return bucket_pos + bucket_size - 1


def make_image_bucket_position(bucket_size, num_relative_distance):
    coords_h = torch.arange(bucket_size)
    coords_w = torch.arange(bucket_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += bucket_size - 1  # shift to start from 0
    relative_coords[:, :, 1] += bucket_size - 1
    relative_coords[:, :, 0] *= 2 * bucket_size - 1
    
    relative_position_index = torch.zeros(size=(bucket_size * bucket_size + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index

class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        seg_embed_tokens,
        no_encoder_attn=False
    ):
        self.args = args
        super().__init__(dictionary) # self.dictionary is registered by the parent
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.decoder_input_type = args.decoder_input_type
        self.tie_seg_projection = resolve_str_true_false(args.tie_seg_projection)
        
        embed_dim = args.decoder_embed_dim
        self.seg_embed_tokens = seg_embed_tokens
        
        num_seg_tokens = args.num_seg_tokens
        self.seg_projection = Linear(embed_dim, num_seg_tokens, bias=False)
        if self.tie_seg_projection:
            logger.info("Tying seg projection weight with seg tokens.")
            self.seg_projection.weight = self.seg_embed_tokens.weight
        
        self.decoder_type = args.decoder_type
        if getattr(args, "decoder_prompt", False):
            self.decoder_prompt_encoder = PromptEncoder(
                type=args.decoder_prompt_type,
                length=args.decoder_prompt_length,
                projection=args.decoder_prompt_projection,
                embed_dim=args.decoder_embed_dim,
                proj_dim=args.decoder_prompt_dim,
                layers=args.decoder_layers,
                vocab_size=args.vocab_size)
            self.decoder_dropout = nn.Dropout(p=0.2)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.num_attention_heads = args.decoder_attention_heads

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.window_size = args.code_image_size // 8

        self.seg_bucket_size=args.patch_image_size // 16
        logger.info(f"seg_bucket_size: {self.seg_bucket_size}")

        self.embed_positions = Embedding(args.max_target_positions + 2, embed_dim)
        self.embed_image_positions = Embedding(args.image_bucket_size ** 2 + 1, embed_dim)
        self.embed_seg_positions = Embedding(self.seg_bucket_size**2 + 1, embed_dim)
        self.pos_ln = LayerNorm(embed_dim)
        self.image_pos_ln = LayerNorm(embed_dim)
        self.seg_pos_ln = LayerNorm(embed_dim)
        
        self.pos_scaling = float(embed_dim / self.num_attention_heads * args.attn_scale_factor) ** -0.5
        self.self_pos_q_linear = nn.Linear(embed_dim, embed_dim)
        self.self_pos_k_linear = nn.Linear(embed_dim, embed_dim)
        self.cross_pos_q_linear = nn.Linear(embed_dim, embed_dim)
        self.cross_pos_k_linear = nn.Linear(embed_dim, embed_dim)

        if getattr(args, "code_layernorm_embedding", False):
            self.code_layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.code_layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        dpr = [x.item() for x in torch.linspace(0, args.decoder_drop_path_rate, args.decoder_layers)]
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn, drop_path_rate=dpr[i])
                for i in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        token_bucket_size = args.token_bucket_size
        token_num_rel_dis = 2 * token_bucket_size - 1
        token_rp_bucket = make_token_bucket_position(token_bucket_size)
        self.token_rel_pos_table_list = nn.ModuleList(
            [Embedding(token_num_rel_dis, self.num_attention_heads, zero_init=True) for _ in range(args.decoder_layers)]
        )

        image_bucket_size = args.image_bucket_size
        image_num_rel_dis = (2 * image_bucket_size - 1) * (2 * image_bucket_size - 1) + 3
        image_rp_bucket = make_image_bucket_position(image_bucket_size, image_num_rel_dis)
        image_position_idx = torch.arange(self.window_size).unsqueeze(0).expand(self.window_size, self.window_size) + \
                            torch.arange(self.window_size).unsqueeze(1) * image_bucket_size + 1
        image_position_idx = torch.cat([torch.tensor([0]), image_position_idx.view(-1)])
        image_position_idx = torch.cat([image_position_idx, torch.tensor([1024] * 769)])
        self.image_rel_pos_table_list = nn.ModuleList(
            [Embedding(image_num_rel_dis, self.num_attention_heads, zero_init=True) for _ in range(args.decoder_layers)]
        )
        
        seg_num_rel_dis = (2 * self.seg_bucket_size - 1) * (2 * self.seg_bucket_size - 1) + 3
        
        self.seg_rel_pos_table_list = nn.ModuleList(
            [Embedding(seg_num_rel_dis, self.num_attention_heads, zero_init=True) for _ in range(args.decoder_layers)]
        )
        seg_rp_bucket = make_image_bucket_position(self.seg_bucket_size, seg_num_rel_dis)
        
        self.register_buffer("seg_rp_bucket", seg_rp_bucket)
        self.register_buffer("token_rp_bucket", token_rp_bucket)
        self.register_buffer("image_rp_bucket", image_rp_bucket)
        self.register_buffer("image_position_idx", image_position_idx)
        
        self.entangle_position_embedding = args.entangle_position_embedding
        
        self.register_buffer("bin_id_offset", torch.tensor([self.dictionary.index("<bin_0>")]))
        self.register_buffer("seg_id_offset", torch.tensor([self.dictionary.index("<seg_0>")]))
        
        self.register_buffer("region_prefix", torch.tensor([976,  35])) # "region: "

    def get_decoder_prompt(self, prompt_tokens):
        past_key_values = self.decoder_prompt_encoder(prompt_tokens)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz,
            seqlen,
            self.args.decoder_layers * 2,
            self.args.decoder_attention_heads,
            self.args.decoder_embed_dim // self.args.decoder_attention_heads,
        )
        past_key_values = self.decoder_dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def output_projection(self, features):
        # Special output projection for supervised segmentation task            
        proj = self.seg_projection(features)
        
        return proj

    def build_decoder_layer(self, args, no_encoder_attn=False, drop_path_rate=0.0):
        layer = TransformerDecoderLayer(args, no_encoder_attn, drop_path_rate= \
            drop_path_rate, use_adapter=getattr(args, "adapter", False), adapter_dim=getattr(args, "adapter_dim", 200))
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def get_rel_pos_bias(self, x, idx):
        seq_len = x.size(1)
        rp_bucket = self.token_rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.token_rel_pos_table_list[idx].weight)
        values = values.permute([2, 0, 1])
        return values.contiguous()

    def get_image_rel_pos_bias(self, x, idx):
        seq_len = x.size(1)
        image_position_idx = self.image_position_idx[:seq_len]
        rp_bucket = self.image_rp_bucket[image_position_idx][:, image_position_idx]
        values = F.embedding(rp_bucket, self.image_rel_pos_table_list[idx].weight)
        values = values.permute(2, 0, 1)
        return values

    def get_seg_rel_pos_bias(self, x, idx):
        seq_len = x.size(1)
        seg_rp_bucket = self.seg_rp_bucket
        values = F.embedding(seg_rp_bucket, self.seg_rel_pos_table_list[idx].weight)
        values = values.permute([2, 0, 1])
        
        return values.contiguous()

    def get_pos_info(self, tokens, tgt_pos_embed, src_pos_embed=None, use_image=False, use_seg=False):
        batch_size = tokens.size(0)
        tgt_len = tokens.size(1)
        
        assert not(use_image and use_seg)
        
        if use_image:
            tgt_pos_embed = self.image_pos_ln(tgt_pos_embed)
        elif use_seg:
            tgt_pos_embed = self.seg_pos_ln(tgt_pos_embed)
        else:
            tgt_pos_embed = self.pos_ln(tgt_pos_embed)
        
        if src_pos_embed is not None:
            src_len = src_pos_embed.size(1)
            pos_q = self.cross_pos_q_linear(tgt_pos_embed).view(
                batch_size, tgt_len, self.num_attention_heads, -1
            ).transpose(1, 2) * self.pos_scaling
            pos_k = self.cross_pos_k_linear(src_pos_embed).view(
                batch_size, src_len, self.num_attention_heads, -1
            ).transpose(1, 2)
        else:
            src_len = tgt_pos_embed.size(1)
            pos_q = self.self_pos_q_linear(tgt_pos_embed).view(
                batch_size, tgt_len, self.num_attention_heads, -1
            ).transpose(1, 2) * self.pos_scaling
            pos_k = self.self_pos_k_linear(tgt_pos_embed).view(
                batch_size, src_len, self.num_attention_heads, -1
            ).transpose(1, 2)
        abs_pos_bias = torch.matmul(pos_q, pos_k.transpose(2, 3))
        
        return abs_pos_bias

    def get_cross_pos_info(self, tokens, tgt_pos_embed, src_pos_embed=None, use_image=False, use_seg=False):
        batch_size = tokens.size(0)
        tgt_len = tokens.size(1)
        
        assert not(use_image and use_seg)
        
        if use_image:
            tgt_pos_embed = self.image_pos_ln(tgt_pos_embed)
        elif use_seg:
            tgt_pos_embed = self.seg_pos_ln(tgt_pos_embed)
        else:
            tgt_pos_embed = self.pos_ln(tgt_pos_embed)
        
        if src_pos_embed is not None:
            bsz, img_len, seq_len, dim = tgt_pos_embed.shape
            pos_q = self.cross_pos_q_linear(tgt_pos_embed).reshape(
                bsz, img_len, seq_len, self.num_attention_heads, -1
            ) * self.pos_scaling
            
            bsz, src_len, dim = src_pos_embed.shape
            pos_k = self.cross_pos_k_linear(src_pos_embed).reshape(
                bsz, src_len, self.num_attention_heads, -1
            )
            abs_pos_bias = torch.einsum('bishd,brhd->bihsr', pos_q, pos_k)
            abs_pos_bias = rearrange(abs_pos_bias, 'b i h s r -> (b i) h s r')
            
        else:
            src_len = tgt_pos_embed.size(1)
            pos_q = self.self_pos_q_linear(tgt_pos_embed).view(
                batch_size, tgt_len, self.num_attention_heads, -1
            ).transpose(1, 2) * self.pos_scaling
            pos_k = self.self_pos_k_linear(tgt_pos_embed).view(
                batch_size, src_len, self.num_attention_heads, -1
            ).transpose(1, 2)
            abs_pos_bias = torch.matmul(pos_q, pos_k.transpose(2, 3))
        
        return abs_pos_bias

    def forward(
        self,
        prev_output_tokens,
        code_masks: Optional[torch.Tensor] = None,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            code_masks=code_masks,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        extra['penultimate'] = x
        if not features_only:
            x = self.output_layer(x)

        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        code_masks: Optional[torch.Tensor],
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        
        if self.decoder_type == 'surrogate':
            extract_features_scriptable = self.extract_features_scriptable_surrogate
        else:
            raise NotImplementedError
            
        return extract_features_scriptable(
            prev_output_tokens,
            code_masks,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable_surrogate(
            self,
            prev_output_tokens,
            code_masks: Optional[torch.Tensor],
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
        ):
            """
            extract_features function that use image tokens as surrogate inputs
            """        
            prompt_tokens = None
            prompt_padding_mask = None
            prompt_kv_list = None
            if self.args.decoder_prompt:
                bsz, seq_len = prev_output_tokens.shape[0], prev_output_tokens.shape[1]
                if self.args.decoder_prompt_type in ("prefix"):
                    prompt_tokens = torch.arange(
                        0, self.args.decoder_prompt_length).to(
                        prev_output_tokens.device)
                    prompt_tokens = prompt_tokens.unsqueeze(0).expand(bsz, -1)
                    prompt_padding_mask = torch.zeros_like(prompt_tokens).to(prompt_tokens.device)
                prompt_kv_list = self.get_decoder_prompt(prompt_tokens)
            bs, slen = prev_output_tokens.size()
            if alignment_layer is None:
                alignment_layer = self.num_layers - 1

            enc: Optional[Tensor] = None
            padding_mask: Optional[Tensor] = None
            if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
                enc = encoder_out["encoder_out"][0]
                assert (
                    enc.size()[1] == bs
                ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
            if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
                padding_mask = encoder_out["encoder_padding_mask"][0]

            bsz = prev_output_tokens.size(0)
            h, w = encoder_out['image_embed_shape'][0]
            tgt_len = h * w + 1
            
            # embed tokens and positions
            bos = prev_output_tokens[:, :1]
            
            if self.decoder_input_type == 'encoder_input':
                image_embed_before_scale = encoder_out["image_embed_before_scale"][0]
            elif self.decoder_input_type == 'encoder_output':
                image_embed_before_scale = rearrange(encoder_out["encoder_out"][0][:h * w], 'l b d -> b l d')

            x = torch.cat([self.embed_tokens(bos), image_embed_before_scale], dim=1)
            x = x * self.embed_scale

            prev_output_tokens = torch.zeros_like(x, dtype=torch.long, device=x.device)
            old_image_position_ids = torch.arange(self.seg_bucket_size).unsqueeze(0).expand(self.seg_bucket_size, self.seg_bucket_size) + \
                                     torch.arange(self.seg_bucket_size).unsqueeze(1) * self.seg_bucket_size + 1
            old_image_position_ids = old_image_position_ids.to(x.device)
            old_image_pos_embed = self.embed_seg_positions(old_image_position_ids)
            old_image_pos_embed = old_image_pos_embed.reshape(1, self.seg_bucket_size, self.seg_bucket_size, -1).permute(0, 3, 1, 2)
            image_pos_embed = F.interpolate(old_image_pos_embed, size=(h, w), mode='bilinear')
            image_pos_embed = image_pos_embed.permute(0, 2, 3, 1).reshape(1, h*w, -1)
            image_pos_embed = image_pos_embed.expand(bsz, -1, -1)
            
            tgt_pos_embed = torch.cat([self.embed_seg_positions(torch.tensor([0],device=x.device)).unsqueeze(0).expand(bsz, -1, -1), image_pos_embed], dim=1)

            # self attn position bias
            self_abs_pos_bias = self.get_pos_info(prev_output_tokens, tgt_pos_embed, use_seg=True)

            # cross attn position bias
            src_pos_embed = encoder_out['position_embeddings'][0]
            cross_abs_pos_bias = self.get_pos_info(prev_output_tokens, tgt_pos_embed, src_pos_embed=src_pos_embed, use_seg=True)
            cross_abs_pos_bias = cross_abs_pos_bias.reshape(-1, *cross_abs_pos_bias.size()[-2:])

            all_prev_output_tokens = prev_output_tokens.clone()
            if incremental_state is not None:
                prev_output_tokens = prev_output_tokens[:, -1:]
                cross_abs_pos_bias = cross_abs_pos_bias[:, -1:, :]
                tgt_pos_embed = tgt_pos_embed[:, -1:, :]
            
            if self.quant_noise is not None:
                x = self.quant_noise(x)

            if self.project_in_dim is not None:
                x = self.project_in_dim(x)

            if self.entangle_position_embedding is not None and not self.args.disable_entangle:
                x += tgt_pos_embed

            if self.layernorm_embedding is not None:
                x = self.layernorm_embedding(x)

            x = self.dropout_module(x)

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

            self_attn_padding_mask: Optional[Tensor] = None
            if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
                self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
                if prompt_padding_mask is not None:
                    self_attn_padding_mask = torch.cat([prompt_padding_mask, self_attn_padding_mask], dim=1)

            # decoder layers
            attn: Optional[Tensor] = None
            inner_states: List[Optional[Tensor]] = [x]
            for idx, layer in enumerate(self.layers):
                if incremental_state is None and not full_context_alignment:
                    self_attn_mask = self.buffered_future_mask(x)
                    if self.args.decoder_prompt:
                        seq_len, prompt_len = x.size(0), prompt_tokens.size(1)
                        prompt_mask = torch.zeros([seq_len, prompt_len]).to(x.device)
                        self_attn_mask = torch.cat([prompt_mask, self_attn_mask], dim=1)
                else:
                    self_attn_mask = None
                self_attn_bias = self_abs_pos_bias.clone()
        
                old_rel_pos_bias = self.get_seg_rel_pos_bias(all_prev_output_tokens, idx).unsqueeze(0)
                old_rel_pos_bias = rearrange(old_rel_pos_bias, 'b c hw1 hw2 -> (b hw2) c hw1')
                old_rel_pos_bias_bos, old_rel_pos_bias_seg = torch.split(old_rel_pos_bias, [1, self.seg_bucket_size**2], dim=-1)

                old_rel_pos_bias_seg = rearrange(old_rel_pos_bias_seg, 'b c (h w) -> b c h w', h=self.seg_bucket_size, w=self.seg_bucket_size)
                
                old_rel_pos_bias_seg = F.interpolate(old_rel_pos_bias_seg, size=(h, w), mode='bilinear')
                
                old_rel_pos_bias_seg = rearrange(old_rel_pos_bias_seg, 'b c h w -> b c (h w)')
                old_rel_pos_bias = torch.cat([old_rel_pos_bias_bos, old_rel_pos_bias_seg], dim=-1)
                
                old_rel_pos_bias = rearrange(old_rel_pos_bias, '(b hw2) c hw1 -> (b hw1) c hw2', hw1=tgt_len, hw2=self.seg_bucket_size**2 + 1)

                old_rel_pos_bias_bos, old_rel_pos_bias_seg = torch.split(old_rel_pos_bias, [1, self.seg_bucket_size**2], dim=-1)
                
                old_rel_pos_bias_seg = rearrange(old_rel_pos_bias_seg, 'b c (h w) -> b c h w', h=self.seg_bucket_size, w=self.seg_bucket_size)
                
                old_rel_pos_bias_seg = F.interpolate(old_rel_pos_bias_seg, size=(h, w), mode='bilinear')
                
                old_rel_pos_bias_seg = rearrange(old_rel_pos_bias_seg, 'b c h w -> b c (h w)')
                rel_pos_bias = torch.cat([old_rel_pos_bias_bos, old_rel_pos_bias_seg], dim=-1)
                
                rel_pos_bias = rearrange(rel_pos_bias, '(b hw1) c hw2 -> b c hw1 hw2', hw1=tgt_len, hw2=tgt_len)
                
                self_attn_bias += rel_pos_bias[..., :prev_output_tokens.size(1), :prev_output_tokens.size(1)]
                
                self_attn_bias = self_attn_bias.reshape(-1, *self_attn_bias.size()[-2:])
                if incremental_state is not None:
                    self_attn_bias = self_attn_bias[:, -1:, :]

                if self.args.decoder_prompt:
                    if self.args.decoder_prompt_type != "prompt":
                        prompt_kv = prompt_kv_list[idx]
                    else:
                        if idx == 0:
                            prompt_kv = prompt_kv_list[idx]
                        else:
                            prompt_kv = None
                else:
                    prompt_kv = None

                x, layer_attn, _ = layer(
                    x,
                    enc,
                    padding_mask,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                    self_attn_bias=self_attn_bias,
                    cross_attn_bias=cross_abs_pos_bias,
                    prompt_kv=prompt_kv
                )
                inner_states.append(x)
                if layer_attn is not None and idx == alignment_layer:
                    attn = layer_attn.float().to(x)

            if attn is not None:
                if alignment_heads is not None:
                    attn = attn[:alignment_heads]

                # average probabilities over heads
                attn = attn.mean(dim=0)

            if self.layer_norm is not None:
                x = self.layer_norm(x)

            # T x B x C -> B x T x C
            x = x.transpose(0, 1)

            if self.project_out_dim is not None:
                x = self.project_out_dim(x)

            return x, {"attn": [attn], "inner_states": inner_states}


    def extract_features_scriptable(
        self,
        prev_output_tokens,
        code_masks: Optional[torch.Tensor],
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        prompt_tokens = None
        prompt_padding_mask = None
        prompt_kv_list = None
        if self.args.decoder_prompt:
            bsz, seq_len = prev_output_tokens.shape[0], prev_output_tokens.shape[1]
            if self.args.decoder_prompt_type in ("prefix"):
                prompt_tokens = torch.arange(
                    0, self.args.decoder_prompt_length).to(
                    prev_output_tokens.device)
                prompt_tokens = prompt_tokens.unsqueeze(0).expand(bsz, -1)
                prompt_padding_mask = torch.zeros_like(prompt_tokens).to(prompt_tokens.device)
            prompt_kv_list = self.get_decoder_prompt(prompt_tokens)
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        bsz, tgt_len = prev_output_tokens.shape
        token_position_idx = utils.new_arange(prev_output_tokens)
        tgt_pos_embed = self.embed_positions(token_position_idx)
        if code_masks is not None and torch.any(code_masks):
            image_position_idx = self.image_position_idx[:prev_output_tokens.size(1)].unsqueeze(0).expand(bsz, tgt_len)
            tgt_pos_embed[code_masks] = self.embed_image_positions(image_position_idx)[code_masks]

        # self attn position bias
        self_abs_pos_bias = self.get_pos_info(prev_output_tokens, tgt_pos_embed, use_image=False)
        if code_masks is not None and torch.any(code_masks):
            self_image_abs_pos_bias = self.get_pos_info(prev_output_tokens, tgt_pos_embed, use_image=True)
            self_abs_pos_bias[code_masks] = self_image_abs_pos_bias[code_masks]
        # cross attn position bias
        src_pos_embed = encoder_out['position_embeddings'][0]
        cross_abs_pos_bias = self.get_pos_info(prev_output_tokens, tgt_pos_embed, src_pos_embed=src_pos_embed)
        if code_masks is not None and torch.any(code_masks):
            cross_image_abs_pos_bias = self.get_pos_info(prev_output_tokens, tgt_pos_embed, src_pos_embed=src_pos_embed, use_image=True)
            cross_abs_pos_bias[code_masks] = cross_image_abs_pos_bias[code_masks]
        cross_abs_pos_bias = cross_abs_pos_bias.reshape(-1, *cross_abs_pos_bias.size()[-2:])

        all_prev_output_tokens = prev_output_tokens.clone()
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            cross_abs_pos_bias = cross_abs_pos_bias[:, -1:, :]
            tgt_pos_embed = tgt_pos_embed[:, -1:, :]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if self.entangle_position_embedding is not None and not self.args.disable_entangle:
            x += tgt_pos_embed

        if self.layernorm_embedding is not None:
            if code_masks is None or not code_masks.any() or not getattr(self, "code_layernorm_embedding", False):
                x = self.layernorm_embedding(x)
            elif code_masks is not None and code_masks.all():
                x = self.code_layernorm_embedding(x)
            else:
                x[~code_masks] = self.layernorm_embedding(x[~code_masks])
                x[code_masks] = self.code_layernorm_embedding(x[code_masks])

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
            if prompt_padding_mask is not None:
                self_attn_padding_mask = torch.cat([prompt_padding_mask, self_attn_padding_mask], dim=1)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
                if self.args.decoder_prompt:
                    seq_len, prompt_len = x.size(0), prompt_tokens.size(1)
                    prompt_mask = torch.zeros([seq_len, prompt_len]).to(x.device)
                    self_attn_mask = torch.cat([prompt_mask, self_attn_mask], dim=1)
            else:
                self_attn_mask = None

            self_attn_bias = self_abs_pos_bias.clone()
            if code_masks is None or not code_masks.any():
                self_attn_bias += self.get_rel_pos_bias(all_prev_output_tokens, idx).unsqueeze(0)
            elif code_masks is not None and code_masks.all():
                self_attn_bias += self.get_image_rel_pos_bias(all_prev_output_tokens, idx).unsqueeze(0)
            else:
                self_attn_bias[~code_masks] += self.get_rel_pos_bias(all_prev_output_tokens, idx).unsqueeze(0)
                self_attn_bias[code_masks] += self.get_image_rel_pos_bias(all_prev_output_tokens, idx).unsqueeze(0)
            self_attn_bias = self_attn_bias.reshape(-1, *self_attn_bias.size()[-2:])
            if incremental_state is not None:
                self_attn_bias = self_attn_bias[:, -1:, :]

            if self.args.decoder_prompt:
                if self.args.decoder_prompt_type != "prompt":
                    prompt_kv = prompt_kv_list[idx]
                else:
                    if idx == 0:
                        prompt_kv = prompt_kv_list[idx]
                    else:
                        prompt_kv = None
            else:
                prompt_kv = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                self_attn_bias=self_attn_bias,
                cross_attn_bias=cross_abs_pos_bias,
                prompt_kv=prompt_kv
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return self.max_target_positions

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        
        if f"{name}.output_projection.weight" in state_dict:
            state_dict.pop(f"{name}.output_projection.weight")
        
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        prefix = name + "." if name != "" else ""
        image_params = ["image_position_idx"]
        for image_param in image_params:
            state_dict[prefix + image_param] = self.state_dict()[image_param]
        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict:
                state_dict[prefix + param_name] = self.state_dict()[param_name]

        if len(state_dict["decoder.embed_image_positions.weight"]) < len(self.state_dict()["embed_image_positions.weight"]):
            num_posids_to_add = len(self.state_dict()["embed_image_positions.weight"]) - len(state_dict["decoder.embed_image_positions.weight"])
            embed_dim = state_dict["decoder.embed_image_positions.weight"].size(1)
            new_pos_embed_to_add = torch.zeros(num_posids_to_add, embed_dim)
            nn.init.normal_(new_pos_embed_to_add, mean=0, std=embed_dim ** -0.5)
            new_pos_embed_to_add = new_pos_embed_to_add.to(
                dtype=state_dict["decoder.embed_image_positions.weight"].dtype,
            )
            state_dict["decoder.embed_image_positions.weight"] = torch.cat(
                [state_dict["decoder.embed_image_positions.weight"], new_pos_embed_to_add]
            )
        if f"{name}.seg_embed_tokens.weight" in state_dict:
            if self.seg_embed_tokens.weight.shape[0] != state_dict[f"{name}.seg_embed_tokens.weight"].shape[0]:
                logger.info(f"Head num_classes mismatch: {self.seg_embed_tokens.weight.shape} vs. {state_dict[f'{name}.seg_embed_tokens.weight'].shape}. Deleting from the checkpoint.")
                del state_dict[f"{name}.seg_embed_tokens.weight"]
                
        if f"{name}.seg_projection.weight" in state_dict:
            if self.seg_projection.weight.shape[0] != state_dict[f"{name}.seg_projection.weight"].shape[0]:
                logger.info(f"Head num_classes mismatch: {self.seg_projection.weight.shape} vs. {state_dict[f'{name}.seg_projection.weight'].shape}. Deleting from the checkpoint.")
                del state_dict[f"{name}.seg_projection.weight"]

        return state_dict
    
class PromptEncoder(torch.nn.Module):
    r"""
    Prompt encoder to generate prompts, including prompt, prefix, instance and instruction
    """

    def __init__(
            self,
            type,
            length,
            projection,
            embed_dim,
            proj_dim,
            layers,
            vocab_size):
        super().__init__()
        self.prefix_projection = projection

        if type == "prefix":
            layers = layers
            prompt_vocab_size = length

        if self.prefix_projection:
            self.embedding = torch.nn.Embedding(prompt_vocab_size, embed_dim)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, proj_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(proj_dim, layers * 2 * embed_dim)
            )
        else:
            if type == "prefix":
                self.embedding = torch.nn.Embedding(
                    prompt_vocab_size, layers * 2 * embed_dim)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
    
def Embedding(num_embeddings, embedding_dim, padding_idx=None, zero_init=False):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    if zero_init:
        nn.init.constant_(m.weight, 0)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x