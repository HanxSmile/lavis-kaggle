# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F


class VitonAttnProcessor(nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, sample_ratio=0.5):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        self.to_w = nn.Linear((hidden_size or cross_attention_dim) + 1, 1)
        self.to_q = nn.Linear(hidden_size or cross_attention_dim, hidden_size, bias=False)
        self.to_k = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.attn_probs = []
        self.src_mask = None
        self.sample_ratio = sample_ratio
        self.condition_flag = None

    def setup_mask(self, src_mask):
        # src_mask.shape = [B, 1, H, W]
        self.attn_probs = []
        self.src_mask = src_mask
        assert self.src_mask.shape[1] == 1

    def setup_condition_flag(self, flag):
        self.condition_flag = flag

    def reshape_mask(self, hidden_states):
        b, seq_len, _ = hidden_states.shape
        if not self.condition_flag:
            seq_len = seq_len // 2
        temp_mask = rearrange(self.src_mask, "b c h w -> b (h w) c").squeeze(-1)

        mask_seq_len = temp_mask.shape[1]
        resize_ratio = int(np.sqrt(mask_seq_len / seq_len))
        _, _, height, width = self.src_mask.shape

        dst_mask = torch.nn.functional.interpolate(
            self.src_mask, size=(height // resize_ratio, width // resize_ratio)
        )
        dst_mask = rearrange(dst_mask, "b c h w -> b (h w) c")

        if not self.condition_flag:
            dst_mask = torch.cat([dst_mask] * 2, dim=1)

        assert dst_mask.shape[1] == hidden_states.shape[
            1], f"dst mask: {dst_mask.shape}\nhidden states: {hidden_states.shape}"
        return dst_mask

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            *args,
            **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        mask = self.reshape_mask(hidden_states)

        weight_net_input = torch.cat([hidden_states, mask], dim=-1)
        weight = torch.sigmoid(self.to_w(weight_net_input))

        query1 = attn.to_q(hidden_states)
        query2 = self.to_q(hidden_states)

        query = query1 * weight + query2 * (1 - weight)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if not self.condition_flag:
            encoder_hidden_states_1, encoder_hidden_states_2 = encoder_hidden_states.chunk(2, dim=1)

            key1 = attn.to_k(encoder_hidden_states_1)
            key2 = self.to_k(encoder_hidden_states_2)
            key = torch.cat([key1, key2], dim=1)

            value1 = attn.to_v(encoder_hidden_states_1)
            value2 = self.to_v(encoder_hidden_states_2)
            value = torch.cat([value1, value2], dim=1)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class VitonAttnProcessor2_0(nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, sample_ratio=0.5):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        self.to_w = nn.Linear((hidden_size or cross_attention_dim) + 1, 1)
        self.to_q = nn.Linear(hidden_size or cross_attention_dim, hidden_size, bias=False)
        self.to_k = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.attn_probs = []
        self.src_mask = None
        self.sample_ratio = sample_ratio
        self.condition_flag = None

    def setup_mask(self, src_mask):
        # src_mask.shape = [B, 1, H, W]
        self.attn_probs = []
        self.src_mask = src_mask
        assert self.src_mask.shape[1] == 1

    def setup_condition_flag(self, flag):
        self.condition_flag = flag

    def reshape_mask(self, hidden_states):
        b, seq_len, _ = hidden_states.shape
        if not self.condition_flag:
            seq_len = seq_len // 2
        temp_mask = rearrange(self.src_mask, "b c h w -> b (h w) c").squeeze(-1)

        mask_seq_len = temp_mask.shape[1]
        resize_ratio = int(np.sqrt(mask_seq_len / seq_len))
        _, _, height, width = self.src_mask.shape

        dst_mask = torch.nn.functional.interpolate(
            self.src_mask, size=(height // resize_ratio, width // resize_ratio)
        )
        dst_mask = rearrange(dst_mask, "b c h w -> b (h w) c")

        if not self.condition_flag:
            dst_mask = torch.cat([dst_mask] * 2, dim=1)

        assert dst_mask.shape[1] == hidden_states.shape[
            1], f"dst mask: {dst_mask.shape}\nhidden states: {hidden_states.shape}"
        return dst_mask

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            *args,
            **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        mask = self.reshape_mask(hidden_states)

        weight_net_input = torch.cat([hidden_states, mask], dim=-1)
        weight = torch.sigmoid(self.to_w(weight_net_input))

        query1 = attn.to_q(hidden_states)
        query2 = self.to_q(hidden_states)

        query = query1 * weight + query2 * (1 - weight)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if not self.condition_flag:
            encoder_hidden_states_1, encoder_hidden_states_2 = encoder_hidden_states.chunk(2, dim=1)

            key1 = attn.to_k(encoder_hidden_states_1)
            key2 = self.to_k(encoder_hidden_states_2)
            key = torch.cat([key1, key2], dim=1)

            value1 = attn.to_v(encoder_hidden_states_1)
            value2 = self.to_v(encoder_hidden_states_2)
            value = torch.cat([value1, value2], dim=1)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(
            self,
            hidden_size=None,
            cross_attention_dim=None,
    ):
        super().__init__()

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            *args,
            **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
            self,
            hidden_size=None,
            cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            *args,
            **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
