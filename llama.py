import json
import pickle

import jax
import jax.numpy as jnp
import haiku as hk

from jaxllama import LlamaModel, LlamaConfig
from jaxllama import get_model

model = get_model('/')