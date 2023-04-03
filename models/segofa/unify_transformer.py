# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import math
from typing import Any, Dict, List, Optional, Tuple
import contextlib

import torch
import torch.nn as nn
from torch import Tensor
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)


from .resnet import ResNet
from .frozen_bn import FrozenBatchNorm2d
from .encoder_module import TransformerEncoder
from .decoder_module import TransformerDecoder

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

def BatchNorm2d(out_chan, momentum=0.1, eps=1e-3):
    return nn.SyncBatchNorm.convert_sync_batchnorm(
        nn.BatchNorm2d(out_chan, momentum=momentum, eps=eps)
    )

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


@register_model("seg_unify_transformer")
class TransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--bitfit', default=False, action='store_true',
                            help='use bitfit in the transformer')



        parser.add_argument('--adapter', action='store_true',
                            help='use adapter in the model')
        parser.add_argument('--adapter-dim', type=int, metavar='N',
                            help='adapter-down-dim')

        parser.add_argument('--encoder-prompt', action='store_true',
                            help='use prompt tuning in the encoder')
        parser.add_argument('--encoder-prompt-type', type=str, metavar='STR',
                            choices=['prefix'],
                            help='the type of prompt tuning')
        parser.add_argument('--encoder-prompt-projection', action='store_true',
                            help='use prompt projection')
        parser.add_argument('--encoder-prompt-length', type=int, metavar='N',
                            help='use prompt tuning in the decoder')
        parser.add_argument('--encoder-prompt-dim', type=int, metavar='N',
                            help='encoder prompt dimension if use encoder prompt projection')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--freeze-decoder', action='store_true',
                            help='freeze the parameters in the decoder')

        parser.add_argument('--decoder-prompt', action='store_true',
                            help='use prompt tuning in the decoder')
        parser.add_argument('--decoder-prompt-type', type=str, metavar='STR',
                            choices=['prefix'],
                            help='the type of prompt tuning')
        parser.add_argument('--decoder-prompt-length', type=int, metavar='N',
                            help='use prompt tuning in the decoder')
        parser.add_argument('--decoder-prompt-projection', action='store_true',
                            help='use prompt projection')
        parser.add_argument('--decoder-prompt-dim', type=int, metavar='N',
                            help='decoder prompt dimension if use decoder prompt projection')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--checkpoint-activations', action='store_true',
                            help='checkpoint activations at each layer, which saves GPU '
                                 'memory usage at the cost of some additional compute')
        parser.add_argument('--offload-activations', action='store_true',
                            help='checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations.')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # args for Fully Sharded Data Parallel (FSDP) training
        parser.add_argument(
            '--min-params-to-wrap', type=int, metavar='D', default=DEFAULT_MIN_PARAMS_TO_WRAP,
            help=(
                'minimum number of params for a layer to be wrapped with FSDP() when '
                'training with --ddp-backend=fully_sharded. Smaller values will '
                'improve memory efficiency, but may make torch.distributed '
                'communication less efficient due to smaller input sizes. This option '
                'is set to 0 (i.e., always wrap) when --checkpoint-activations or '
                '--offload-activations are passed.'
            )
        )

        parser.add_argument('--resnet-drop-path-rate', type=float,
                            help='resnet drop path rate')
        parser.add_argument('--encoder-drop-path-rate', type=float,
                            help='encoder drop path rate')
        parser.add_argument('--decoder-drop-path-rate', type=float,
                            help='encoder drop path rate')

        parser.add_argument('--token-bucket-size', type=int,
                            help='token bucket size')
        parser.add_argument('--image-bucket-size', type=int,
                            help='image bucket size')

        parser.add_argument('--attn-scale-factor', type=float,
                            help='attention scale factor')
        parser.add_argument('--freeze-resnet', type=str, default='false', help='freeze resnet (bn only)')
        parser.add_argument('--freeze-entire-resnet', type=str, default='false', help='freeze resnet (all params)')
        parser.add_argument('--freeze-encoder-embedding', type=str, default='false', help='freeze encoder token embedding')
        parser.add_argument('--freeze-decoder-embedding', type=str, default='false',  help='freeze decoder token embedding')
        parser.add_argument('--freeze-seg-embedding', type=str, default='false',  help='freeze seg token embedding')
        parser.add_argument('--freeze-encoder-transformer', type=str, default='false', help='freeze the parameters in the encoder transformer')
        parser.add_argument('--freeze-encoder-transformer-layers', type=int, default=0, help='freeze the parameters in the encoder transformer')
        parser.add_argument('--add-type-embedding', action='store_true',
                            help='add source/region/patch type embedding')
        parser.add_argument('--interpolate-position', action='store_true',
                            help='interpolate position')

        parser.add_argument('--resnet-type', choices=['resnet50', 'resnet101', 'resnet152'],
                            help='resnet type')
        parser.add_argument('--resnet-model-path', type=str, metavar='STR',
                            help='path to load resnet')
        parser.add_argument('--code-image-size', type=int,
                            help='code image size')
        parser.add_argument('--patch-layernorm-embedding', action='store_true',
                            help='add layernorm to patch embedding')
        parser.add_argument('--code-layernorm-embedding', action='store_true',
                            help='add layernorm to code embedding')
        parser.add_argument('--entangle-position-embedding', action='store_true',
                            help='entangle position embedding')
        parser.add_argument('--disable-entangle', action='store_true',
                            help='disable entangle')
        parser.add_argument('--sync-bn', action='store_true',
                            help='sync batchnorm')

        parser.add_argument('--scale-attn', action='store_true',
                            help='scale attn')
        parser.add_argument('--scale-fc', action='store_true',
                            help='scale fc')
        parser.add_argument('--scale-heads', action='store_true',
                            help='scale heads')
        parser.add_argument('--scale-resids', action='store_true',
                            help='scale resids')        
        parser.add_argument('--num-seg-tokens', type=int, default=150,
                            help='Number of segmentation tokens')
        parser.add_argument('--decoder-type', type=str, default='surrogate',
                            help='surrogate')
        parser.add_argument('--tie-seg-projection', type=str, default='false',
                            help='Whether to tie the seg projection weights with seg tokens')
        
        parser.add_argument('--decoder-input-type', type=str, default='encoder_input',
                            help='encoder_input | encoder_output')
        
        parser.add_argument('--patch-image-size', type=int, default=512,
                            help='patch_image_size')
        parser.add_argument('--orig-patch-image-size', type=int, default=512,
                            help='patch_image_size')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path, seg_embed_tokens
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path, seg_embed_tokens
            )
        
        seg_embed_tokens = Embedding(args.num_seg_tokens, args.encoder_embed_dim)
        
        freeze_encoder_embedding = resolve_str_true_false(args.freeze_encoder_embedding)
        freeze_decoder_embedding = resolve_str_true_false(args.freeze_decoder_embedding)
        freeze_seg_embedding = resolve_str_true_false(args.freeze_seg_embedding)
        
        if freeze_encoder_embedding or getattr(
                args, "encoder_prompt", False) or getattr(args, "decoder_prompt", False) or getattr(args, "adapter", False):    
            encoder_embed_tokens.weight.requires_grad = False
        if freeze_decoder_embedding or getattr(
                args, "encoder_prompt", False) or getattr(args, "decoder_prompt", False) or getattr(args, "adapter", False):    
            decoder_embed_tokens.weight.requires_grad = False
        if freeze_seg_embedding:
            seg_embed_tokens.weight.requires_grad = False
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens, seg_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens, seg_embed_tokens)
        if getattr(args, "encoder_prompt", False) or getattr(
                args, "decoder_prompt", False):
            encoder.requires_grad_(False)
            decoder.requires_grad_(False)
            if getattr(args, "encoder_prompt", False):
                encoder.encoder_prompt_encoder.requires_grad_(True)
            if getattr(args, "decoder_prompt", False):
                decoder.decoder_prompt_encoder.requires_grad_(True)
            if getattr(args, "adapter", False):
                for idx, layer in enumerate(encoder.layers):
                    layer.adapter.requires_grad_(True)
                for idx, layer in enumerate(decoder.layers):
                    layer.adapter.requires_grad_(True)        
        if not args.share_all_embeddings:
            min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)
        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary) - args.num_seg_tokens
        padding_idx = dictionary.pad()

        args.vocab_size = num_embeddings
        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens, seg_embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens, seg_embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, seg_embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            seg_embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        full_context_alignment: bool = False,
        text2seg_decoding: bool = False,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            text2seg_decoding=text2seg_decoding
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

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


@register_model_architecture("seg_unify_transformer", "seg_unify_transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.encoder_prompt = getattr(args, "encoder_prompt", False)
    args.encoder_prompt_length = getattr(args, "encoder_prompt_length", 100)
    args.encoder_prompt_type = getattr(args, "encoder_prompt_type", "prefix")
    args.encoder_prompt_projection = getattr(args, "encoder_prompt_projection", False)
    args.encoder_prompt_dim = getattr(args, "encoder_prompt_dim", 2 * args.encoder_embed_dim)

    args.decoder_prompt = getattr(args, "decoder_prompt", False)
    args.decoder_prompt_length = getattr(args, "decoder_prompt_length", 100)
    args.decoder_prompt_type = getattr(args, "decoder_prompt_type", "prefix")
    args.decoder_prompt_projection = getattr(args, "decoder_prompt_projection", False)
    args.decoder_prompt_dim = getattr(args, "decoder_prompt_dim", 2 * args.encoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)
