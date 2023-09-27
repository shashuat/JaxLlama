from .config import ConfigModule

class LlamaRotaryEmbedding(ConfigModule):
    def __init__(self, config, base=10000):
        super().__init__(config=config)
        self.base = base
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        dtype = get_dtype()

        inv_freq = 1 / (self.base ** (jnp.arange(0, head_dim, 2.0, dtype=jnp.float32) / head_dim))
        freqs = jnp.arange(self.config.max_position_embeddings)[:, None] * inv_freq[None, :]
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        self.sin = jnp.sin(emb).astype(dtype)
        self.cos = jnp.cos(emb).astype(dtype)

    def __call__(self, queries, keys, pos_ids):
        cos = self.cos[pos_ids]
        sin = self.sin[pos_ids]
        q_emb = queries * cos + rotate_half(queries) * sin
        k_emb = keys * cos + rotate_half(keys) * sin
        return q_emb, k_emb