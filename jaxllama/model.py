"""Adapted from torch implementation at https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py"""
from collections import namedtuple
from functools import partial
import logging
import warnings

import haiku as hk
from jax.ad_checkpoint import checkpoint_name
from .attention import LlamaAttention

logger = logging.getLogger(__name__)

LlamaConfig = namedtuple('LlamaConfig', [
    'vocab_size',
    'hidden_size',
    'intermediate_size',
    'max_position_embeddings',
    'num_attention_heads',
    'rms_norm_eps',
    'num_hidden_layers',
])

def get_dtype():
    policy = hk.mixed_precision.current_policy()
    return policy.compute_dtype if policy else jnp.float32


class VarianceOnlyLayerNorm(ConfigModule):
    """LayerNorm but without subtracting the mean"""
    def __call__(self, x):
        variance = jnp.mean((x.astype(jnp.float32) ** 2), axis=-1, keepdims=True)
        scale = hk.get_parameter(
            'weight',
            x.shape[-1:],
            init=hk.initializers.Constant(1.)
        )
        x = x * jax.lax.rsqrt(variance + self.config.rms_norm_eps)
        return (x * scale).astype(get_dtype())

def rotate_half(x):
    n = x.shape[-1] // 2
    x1 = x[..., :n]
    x2 = x[..., n:]
    return jnp.concatenate([-x2, x1], axis=-1)

class LlamaMLP(ConfigModule):
    def __call__(self, x):
        gate = jax.nn.silu(hk.Linear(self.config.intermediate_size, name='gate_proj', with_bias=False)(x))
        val = hk.Linear(self.config.intermediate_size, name='up_proj', with_bias=False)(x)

        return hk.Linear(self.config.hidden_size, name='down_proj', with_bias=False)(gate * val)

def _make_causal_mask(input_indices, past_cache_size=None):
    seq_len = input_indices.shape[-1]
    kv_length = seq_len if past_cache_size is None else past_cache_size
    mask = jnp.full((seq_len, kv_length), -jnp.inf)
    mask = jnp.where(
        input_indices[:, None] >= jnp.arange(kv_length)[None, :],
        0.,
        mask
    )
    return mask

class LlamaModel(ConfigModule):
    def __init__(self, config, name='model'):
        super().__init__(config=config)

    def __call__(
        self,
        input_ids,
        past=None,
        past_cache_size=None,
        return_past=False,
        return_hidden=False,
        checkpoint=False,
        checkpoint_mlp=False,
        mlp_block_size=1,
        use_flash_attention=False,
        no_cache_update=False,
    ):
        inp_length = input_ids.shape[-1]
        if past is None:
            past_length = 0
            if past_cache_size is not None:
                cache_shape = (
                    2,
                    self.config.num_attention_heads,
                    past_cache_size,
                    self.config.hidden_size // self.config.num_attention_heads
                )
                past = [
                    jnp.zeros(
                        cache_shape,
                        dtype=get_dtype()
                     )
                    for _ in range(self.config.num_hidden_layers)
                ]
            else:
                past = [None] * self.config.num_hidden_layers
                past_cache_size = 0
            indices = jnp.arange(inp_length)
        else:
            past, past_length = past
            if past_cache_size != past[0][0].shape[1]:
                logger.warning(f'past_cache_size {past_cache_size} != {past[0][0].shape[1]}, passed value will be ignored')
                past_cache_size = past[0][0].shape[1]

            indices = jax.lax.dynamic_slice_in_dim(jnp.arange(past_cache_size), past_length, inp_length)

        if no_cache_update:
            attention_mask = _make_causal_mask(
                jnp.arange(inp_length),
                past_cache_size=None,
            )
        else:
            attention_mask = _make_causal_mask(
                indices,
                past_cache_size=past_cache_size if past_cache_size else None,
            )

        full_seq_length = inp_length + past_length
        wte = hk.get_parameter(
            'embed_tokens_weight',
            shape=(self.config.vocab_size, self.config.hidden_size),
            init=hk.initializers.RandomUniform(-0.02, 0.02)
        )
        hidden_states = wte[input_ids,]

        pos_emb = LlamaRotaryEmbedding(config=self.config)

        presents = []
        hidden = []
        for layer_num, layer_past in enumerate(past):
            if return_hidden:
                hidden.append(hidden_states)
            layer = LLamaDecoderLayer(
                config=self.config,
                pos_emb=pos_emb,
                name=f'layer_{layer_num}',
                checkpoint_mlp=checkpoint_mlp,
                mlp_block_size=mlp_block_size,
                use_flash_attention=use_flash_attention
            )
            layer = partial(layer, no_cache_update=no_cache_update)
            if checkpoint:
                layer = hk.remat(
                    layer,
                    static_argnums=(3,),
                )
            hidden_states, present = layer(
                hidden_states,
                attention_mask,
                indices,
                (layer_past, past_length),
            )
            hidden_states = jax.ad_checkpoint.checkpoint_name(hidden_states, f'llama_hidden_state_{layer_num}')
            if return_past:
                presents.append(present)

        norm_out = VarianceOnlyLayerNorm(self.config, name='norm')(hidden_states)

        ret = {}

        if return_hidden:
            hidden.append(norm_out)
            ret['hidden'] = hidden


        logits = hk.Linear(self.config.vocab_size, name='lm_head', with_bias=False)(norm_out)
        ret['logits'] = logits

        if return_past:
            ret['past'] = (presents, full_seq_length)

        return ret