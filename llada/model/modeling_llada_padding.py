# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

# Copyright 2025 Xinhua Chen
# 
# This file has been modified by Xinhua Chen. Changes include:
# 1. Modified the RoPE implementation to handle discontinuous position IDs.
# 2. Added fake key padding with RoPE instead of suffix slicing.

from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)
from dataclasses import fields
from typing import List, Optional, Tuple, Union
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTN_AVAILABLE = True
except:
    FLEX_ATTN_AVAILABLE = False
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel
from transformers.cache_utils import Cache

from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
    ActivationCheckpointingStrategy,
)
from einops import rearrange

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")

# Import all necessary classes from the original modeling_llada
from .modeling_llada import (
    scaled_dot_product_attention,
    ModuleType,
    init_weights,
    ensure_finite_,
    activation_checkpoint_function,
    BufferCache,
    _non_meta_init_device,
    Dropout,
    LayerNormBase,
    LayerNorm,
    RMSLayerNorm,
    GemmaRMSLayerNorm,
    RotaryEmbedding,
    Activation,
    GELU,
    ReLU,
    SiLU,
    SwiGLU,
    causal_attention_bias,
    get_causal_attention_bias,
    alibi_attention_bias,
    LLaDABlock,
    LLaDAOutput,
    create_model_config_from_pretrained_config,
    LLaDAModelLM,
)

__all__ = [
    "LLaDALlamaBlockPadding",
    "LLaDAModelPadding",
    "LLaDAModelLMPadding",
]

log = logging.getLogger(__name__)


class LLaDALlamaBlockPadding(LLaDABlock):
    """
    LLaDA Llama Block with fake key padding instead of suffix slicing.
    This block pads fake keys with RoPE instead of cutting the suffix.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache, mask_token_id: int = 126336):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache
        self.mask_token_id = mask_token_id

        # Attention input projection. Projects x -> (q, k, v)
        head_dim = config.d_model // config.n_heads
        q_proj_out_dim = config.d_model
        k_proj_out_dim = config.effective_n_kv_heads * head_dim
        v_proj_out_dim = config.effective_n_kv_heads * head_dim
        self.q_proj = nn.Linear(
            config.d_model, q_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.k_proj = nn.Linear(
            config.d_model, k_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.v_proj = nn.Linear(
            config.d_model, v_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )
        self.up_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

        # Fake key base vector - will be initialized from MASK token embedding
        # This is registered as a buffer that will be set from the model's embedding layer
        self.register_buffer('fake_key_base', None)
        self.fake_key_base_initialized = False

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.up_proj, d=self.config.d_model, layer_id=None)

    def _initialize_fake_key_base(self, wte_embedding: nn.Embedding):
        """
        Initialize fake key base from MASK token embedding.
        This should be called once after model initialization.
        """
        if not self.fake_key_base_initialized:
            with torch.no_grad():
                mask_emb = wte_embedding(torch.tensor([self.mask_token_id], device=wte_embedding.weight.device))
                # Project through k_proj to get the right dimension
                # We need to get the dimension for key: effective_n_kv_heads * head_dim
                head_dim = self.config.d_model // self.config.n_heads
                k_dim = self.config.effective_n_kv_heads * head_dim
                # Use a simple linear projection or just use the embedding directly
                # For now, we'll use the embedding and project it
                fake_base = mask_emb.squeeze(0)  # (d_model,)
                # Store as buffer (requires_grad=False)
                self.register_buffer('fake_key_base', fake_base.clone())
                self.fake_key_base_initialized = True

    def _create_fake_keys(
        self,
        real_k: torch.Tensor,
        real_v: torch.Tensor,
        real_k_len: int,
        block_end: int,
        total_seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create fake keys and values for the far suffix (after block_end).
        
        Args:
            real_k: Real keys, shape (B, n_kv_h, real_k_len, hs)
            real_v: Real values, shape (B, n_kv_h, real_k_len, hs)
            real_k_len: Length of real keys (current input length)
            block_end: End position of the current block (absolute position)
            total_seq_len: Total sequence length
            device: Device to create tensors on
            dtype: Data type
            
        Returns:
            k_with_fake: Keys with fake keys concatenated, shape (B, n_kv_h, real_k_len + fake_len, hs)
            v_with_fake: Values with fake values (zeros) concatenated, shape (B, n_kv_h, real_k_len + fake_len, hs)
        """
        B, n_kv_h, _, hs = real_k.shape
        
        # Calculate far suffix length (from block_end to total_seq_len)
        fake_len = total_seq_len - block_end
        if fake_len <= 0:
            # No fake keys needed
            return real_k, real_v
        
        # Initialize fake key base if not done
        if self.fake_key_base is None:
            # Fallback to zeros if not initialized
            fake_key_base = torch.zeros(self.config.d_model, device=device, dtype=dtype)
        else:
            fake_key_base = self.fake_key_base.to(device).to(dtype)
        
        # Create fake keys: (fake_len, d_model)
        # Create fake keys: optimize by projecting once then expanding
        fake_key_projected = self.k_proj(fake_key_base)  # (d_model) -> (k_dim = n_kv_h * hs)
        fake_key_projected = fake_key_projected.view(n_kv_h, hs)  # (n_kv_h, hs)
        fake_keys = fake_key_projected.unsqueeze(0).unsqueeze(2).expand(B, -1, fake_len, -1) 
        # (B, n_kv_h, fake_len, hs)
        
        # Create fake values: zeros (B, n_kv_h, fake_len, hs)
        fake_values = torch.zeros(B, n_kv_h, fake_len, hs, device=device, dtype=dtype)
        
        # Concatenate real and fake
        k_with_fake = torch.cat([real_k, fake_keys], dim=-2)  # (B, n_kv_h, real_k_len + fake_len, hs)
        v_with_fake = torch.cat([real_v, fake_values], dim=-2)  # (B, n_kv_h, real_k_len + fake_len, hs)
        
        return k_with_fake, v_with_fake

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        replace_position: Optional[torch.Tensor] = None,
        q_indices: Optional[torch.Tensor] = None,
        k_indices: Optional[torch.Tensor] = None,
        update_rope: Optional[bool] = False,
        seq_len: Optional[int] = None,
        block_end: Optional[int] = None,
        block_start: Optional[int] = None,
        total_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Attention with fake key padding.
        
        Args:
            block_end: End position of current block (absolute position in full sequence)
            block_start: Start position of current block (absolute position in full sequence)
            total_seq_len: Total sequence length including far suffix
        """
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype
        device = k.device

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # shape: (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        # Handle past key values
        if layer_past is not None: 
            past_key, past_value = layer_past
            if replace_position is None:
                k = torch.cat((past_key, k), dim=-2)
                v = torch.cat((past_value, v), dim=-2)
            else:
                replace_indices = replace_position.nonzero(as_tuple=True)[1]
                past_key[:, :, replace_indices] = k
                k = past_key
                past_value[:, :, replace_indices] = v
                v = past_value

        # Store real key length before adding fake keys
        real_k_len = k.shape[-2]
        
        # Create fake keys for far suffix if needed
        if block_end is not None and total_seq_len is not None and total_seq_len > block_end:
            k, v = self._create_fake_keys(k, v, real_k_len, block_end, total_seq_len, device, dtype)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]

        # Apply RoPE
        if self.config.rope:
            # For fake keys, we need to apply RoPE with absolute position indices
            # Real keys: positions 0 to real_k_len-1 (current input positions)
            # Fake keys: positions block_end to total_seq_len-1 (far suffix positions)
            if block_end is not None and total_seq_len is not None and total_seq_len > block_end:
                # We need to apply RoPE with absolute positions
                # Get RoPE embeddings for the full sequence
                pos_sin, pos_cos = self.rotary_emb.get_rotary_embedding(total_seq_len, device)
                pos_sin = pos_sin.type_as(q)
                pos_cos = pos_cos.type_as(q)
                
                # Apply RoPE to queries
                # Query positions should be relative to block_end (the end of current block)
                if query_len <= block_end:
                    q_pos_sin = pos_sin[:, :, block_end - query_len : block_end, :]
                    q_pos_cos = pos_cos[:, :, block_end - query_len : block_end, :]
                else:
                    # Query extends beyond block_end, use positions up to query_len
                    q_pos_sin = pos_sin[:, :, :query_len, :]
                    q_pos_cos = pos_cos[:, :, :query_len, :]
                
                q = self.rotary_emb.apply_rotary_pos_emb(q_pos_sin, q_pos_cos, q)
                
                # Apply RoPE to keys
                # Real keys: use positions corresponding to their actual positions in the sequence
                k_real = k[:, :, :real_k_len, :]
                k_fake = k[:, :, real_k_len:, :]
                
                # Determine real keys' start position
                if block_start is not None:
                    # If block_start is provided, use it
                    k_real_start_pos = block_start
                else:
                    # Otherwise, infer from block_end and real_k_len
                    k_real_start_pos = max(0, block_end - real_k_len)
                
                # RoPE for real keys: positions from k_real_start_pos to (k_real_start_pos + real_k_len - 1)
                # But we need to make sure we don't exceed block_end
                k_real_end_pos = min(block_end, k_real_start_pos + real_k_len)
                k_real_pos_sin = pos_sin[:, :, k_real_start_pos:k_real_end_pos, :]
                k_real_pos_cos = pos_cos[:, :, k_real_start_pos:k_real_end_pos, :]
                # If the length doesn't match, we need to handle it
                if k_real_pos_sin.shape[2] != real_k_len:
                    # Pad or truncate as needed
                    if k_real_pos_sin.shape[2] < real_k_len:
                        # Need to pad (shouldn't happen normally)
                        pad_len = real_k_len - k_real_pos_sin.shape[2]
                        pad_sin = torch.zeros_like(k_real_pos_sin[:, :, :1, :]).expand(-1, -1, pad_len, -1)
                        pad_cos = torch.ones_like(k_real_pos_cos[:, :, :1, :]).expand(-1, -1, pad_len, -1)
                        k_real_pos_sin = torch.cat([k_real_pos_sin, pad_sin], dim=2)
                        k_real_pos_cos = torch.cat([k_real_pos_cos, pad_cos], dim=2)
                    else:
                        k_real_pos_sin = k_real_pos_sin[:, :, :real_k_len, :]
                        k_real_pos_cos = k_real_pos_cos[:, :, :real_k_len, :]
                k_real = self.rotary_emb.apply_rotary_pos_emb(k_real_pos_sin, k_real_pos_cos, k_real)
                
                # RoPE for fake keys with absolute positions (block_end to total_seq_len-1)
                fake_k_len = total_seq_len - block_end
                fake_pos_sin = pos_sin[:, :, block_end:total_seq_len, :]
                fake_pos_cos = pos_cos[:, :, block_end:total_seq_len, :]
                k_fake = self.rotary_emb.apply_rotary_pos_emb(fake_pos_sin, fake_pos_cos, k_fake)
                
                k = torch.cat([k_real, k_fake], dim=-2)
            else:
                # No fake keys, use standard RoPE
                if q_indices is not None:
                    q, k = self.rotary_emb(q, k, q_indices=q_indices, k_indices=k_indices, 
                                          update=update_rope, seq_len=seq_len, block_end=block_end)
                else:
                    query_len, key_len = q.shape[-2], k.shape[-2]
                    pos_sin, pos_cos = self.rotary_emb.get_rotary_embedding(key_len, device)
                    pos_sin = pos_sin.type_as(q)
                    pos_cos = pos_cos.type_as(q)
                    if block_end is None:
                        q = self.rotary_emb.apply_rotary_pos_emb(
                            pos_sin[:, :, key_len - query_len : key_len, :],
                            pos_cos[:, :, key_len - query_len : key_len, :],
                            q,
                        )
                    else:
                        q = self.rotary_emb.apply_rotary_pos_emb(
                            pos_sin[:, :, block_end - query_len : block_end, :],
                            pos_cos[:, :, block_end - query_len : block_end, :],
                            q,
                        )
                    k = self.rotary_emb.apply_rotary_pos_emb(pos_sin, pos_cos, k)

        if attention_bias is not None:
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # Get the attention scores.
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=False,
        )
        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        return self.attn_out(att), present

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        replace_position: Optional[torch.Tensor] = None,
        q_indices: Optional[torch.Tensor] = None,
        k_indices: Optional[torch.Tensor] = None,
        update_rope: Optional[bool] = False,
        seq_len: Optional[int] = None,
        block_end: Optional[int] = None,
        block_start: Optional[int] = None,
        total_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)
        
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(
                self.attention, q, k, v, attention_bias, layer_past=layer_past, 
                use_cache=use_cache, replace_position=replace_position,
                q_indices=q_indices, k_indices=k_indices, update_rope=update_rope,
                seq_len=seq_len, block_end=block_end, block_start=block_start, total_seq_len=total_seq_len
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, 
                                        layer_past=layer_past, 
                                        use_cache=use_cache,
                                        replace_position=replace_position,  
                                        q_indices=q_indices,
                                        k_indices=k_indices,
        update_rope=update_rope,
        seq_len=seq_len,
        block_end=block_end,
        block_start=block_start,
        total_seq_len=total_seq_len)

        x = x + self.dropout(att)
        og_x = x

        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)
        else:
            x = self.ff_norm(x)
        x, x_up = self.ff_proj(x), self.up_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)
        else:
            x = self.act(x)
        x = x * x_up
        x = self.ff_out(x)
        x = self.dropout(x)

        x = og_x + x

        return x, cache


# Import LLaDAModel base class and create padding version
from .modeling_llada import LLaDAModel

class LLaDAModelPadding(LLaDAModel):
    """
    LLaDA Model with fake key padding support.
    """
    
    def __init__(self, config: ModelConfig, init_params: bool = True, mask_token_id: int = 126336):
        # Temporarily replace block builder to use padding blocks
        original_build = LLaDABlock.build
        
        def build_padding_block(layer_id: int, config: ModelConfig, cache: BufferCache) -> LLaDABlock:
            return LLaDALlamaBlockPadding(layer_id, config, cache, mask_token_id=mask_token_id)
        
        LLaDABlock.build = staticmethod(build_padding_block)
        
        try:
            super().__init__(config, init_params=init_params)
        finally:
            # Restore original builder
            LLaDABlock.build = staticmethod(original_build)
        
        self.mask_token_id = mask_token_id
        
        # Initialize fake key bases for all blocks
        self._initialize_fake_key_bases()
    
    def _initialize_fake_key_bases(self):
        """Initialize fake key bases from MASK token embedding."""
        for block in self.transformer.blocks:
            if isinstance(block, LLaDALlamaBlockPadding):
                block._initialize_fake_key_base(self.transformer.wte)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        replace_position: Optional[torch.Tensor] = None,
        q_indices: Optional[torch.Tensor] = None,
        k_indices: Optional[torch.Tensor] = None,
        update_rope: Optional[bool] = False,
        total_seq_len: Optional[int] = None,
        block_end: Optional[int] = None,
        block_start: Optional[int] = None,
    ) -> LLaDAOutput:
        """
        Forward pass with block_end and total_seq_len for fake key padding.
        """
        # Call parent forward but pass block_end and total_seq_len to blocks
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings

        if self.config.input_emb_norm:
            x = x * (self.config.d_model**0.5)

        if not (self.config.alibi or self.config.rope):
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            pos_emb = self.transformer.wpe(pos)
            x = pos_emb + x

        x = self.transformer.emb_drop(x)

        if attention_mask is not None and 0.0 in attention_mask:
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
        else:
            attention_mask = None

        if (
            attention_bias is not None
            or attention_mask is not None
            or self.config.alibi
            or past_key_values is not None
        ):
            if attention_bias is None and self.config.alibi:
                attention_bias = get_causal_attention_bias(
                    self._LLaDAModel__cache, past_length + seq_len, x.device
                ) + self.get_alibi_attention_bias(past_length + seq_len, x.device)
            elif attention_bias is None:
                # Use name mangled cache attribute from parent class
                cache = getattr(self, '_LLaDAModel__cache', None)
                if cache is None:
                    # Fallback: try to get cache from parent
                    cache = self.__dict__.get('_LLaDAModel__cache')
                attention_bias = get_causal_attention_bias(cache, past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        all_hidden_states = []

        assert self.config.block_group_size == 1
        assert len(self.transformer.blocks) > 0
        for block_idx, block in enumerate(self.transformer.blocks):
            if output_hidden_states:
                all_hidden_states.append(x)

            layer_past = None if past_key_values is None else past_key_values[block_idx]
            update_rope = update_rope & (block_idx == 0)

            x, cache = block(x, 
                             attention_bias=attention_bias, 
                             layer_past=layer_past, 
                             use_cache=use_cache,
                             replace_position=replace_position, 
                             q_indices=q_indices, 
                             k_indices=k_indices,
        update_rope=update_rope,
        seq_len=total_seq_len,
        block_end=block_end,
        block_start=block_start,
        total_seq_len=total_seq_len)

            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)

        if last_logits_only:
            x = x[:, -1, :].unsqueeze(1)

        x = self.transformer.ln_f(x)
        if output_hidden_states:
            all_hidden_states.append(x)

        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)
        else:
            logits = self.transformer.ff_out(x)
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        return LLaDAOutput(logits=logits, attn_key_values=attn_key_values, hidden_states=tuple(all_hidden_states) if output_hidden_states else None)


class LLaDAModelLMPadding(PreTrainedModel):
    """
    HuggingFace wrapper for LLaDA Model with fake key padding.
    """
    
    config_class = LLaDAConfig
    base_model_prefix = "model"
    _no_split_modules = ["LLaDABlock", "LLaDASequentialBlock", "LLaDALlamaBlock", "LLaDALlamaBlockPadding"]

    def __init__(self, config: LLaDAConfig, model: Optional[LLaDAModelPadding] = None, init_params: bool = False, mask_token_id: int = 126336):
        super().__init__(config)

        if not model:
            model_config = create_model_config_from_pretrained_config(config)
            model_config.init_device = "cpu"
            self.model = LLaDAModelPadding(model_config, init_params=init_params, mask_token_id=mask_token_id)
        else:
            self.model = model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        replace_position: Optional[torch.Tensor] = None,
        q_indices: Optional[torch.Tensor] = None,
        k_indices: Optional[torch.Tensor] = None,
        update_rope: Optional[bool] = False,
        seq_len: Optional[int] = None,
        block_end: Optional[int] = None,
        block_start: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if use_cache is None:
            use_cache = self.config.use_cache

        if output_attentions:
            raise ValueError("output_attentions is not yet supported in LLaDA")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            replace_position=replace_position,
            q_indices=q_indices,
            k_indices=k_indices,
            update_rope=update_rope,
            total_seq_len=seq_len,
            block_end=block_end,
            block_start=block_start,
        )
        
        logits = outputs.logits
        hidden_states = outputs.hidden_states

        loss = None
        if labels is not None:
            import warnings
            warnings.warn("Note that for LLaDA, you cannot calculate the loss here.", UserWarning)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
        )

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple]] = None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}
        model_inputs.update(kwargs)
        model_inputs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
        return model_inputs

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module):
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        if self.config.weight_tying:
            return self.model.transformer.wte
        else:
            return self.model.transformer.ff_out

    def set_output_embeddings(self, value: torch.nn.Module):
        if self.config.weight_tying:
            self.model.transformer.wte = value
        else:
            self.model.transformer.ff_out = value

    def tie_weights(self):
        if self.config.weight_tying:
            self.model.transformer.ff_out = self.model.transformer.wte

