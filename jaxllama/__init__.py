from functools import partial
import logging
import warnings

from flash_attention_jax import causal_flash_attention
import haiku as hk
import jax
import jax.numpy as jnp
import jmp

from .model import LlamaModel, LlamaConfig
from .load import get_generator, get_model, load_config, load_weights
from .utils import simple_dtype_policy
from .config import ConfigModule
from .decoder import LLamaDecoderLayer
