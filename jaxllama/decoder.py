from .config import ConfigModule



class LLamaDecoderLayer(ConfigModule):
    def __init__(self, pos_emb, checkpoint_mlp=False, mlp_block_size=None, use_flash_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = pos_emb
        self.checkpoint_mlp = checkpoint_mlp
        self.mlp_block_size = mlp_block_size
        self.use_flash_attention = use_flash_attention

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past,
        **kwargs
    ):
        residual = hidden_states

        norm_output = VarianceOnlyLayerNorm(self.config, name='input_layernorm')(hidden_states)

        attention_layer = LlamaAttention(
            pos_emb=self.pos_emb,
            config=self.config,
            name='self_attn',
            use_flash_attention=self.use_flash_attention
        )
        if self.checkpoint_mlp:
            attention_layer = hk.remat(attention_layer)

        attn_output, present = attention_layer(
            norm_output,
            attention_mask,
            position_ids,
            past,
            **kwargs
        )
        residual += attn_output

        post_attn_norm_output = VarianceOnlyLayerNorm(self.config, name='post_attention_layernorm')(residual)
        mlp_layer = LlamaMLP(self.config)
        if self.checkpoint_mlp:
            mlp_layer = hk.remat(mlp_layer)

        if self.mlp_block_size is None:
            mlp_output = mlp_layer(post_attn_norm_output)
        elif post_attn_norm_output.shape[0] % self.mlp_block_size == 0:
            n_blocks = post_attn_norm_output.shape[0] // self.mlp_block_size
            inps = jnp.split(post_attn_norm_output, n_blocks, axis=0)
            outs = []
            for inp in inps:
                outs.append(mlp_layer(inp))
            mlp_output = jnp.concatenate(outs, axis=0)
        else:
            raise ValueError(f'MLP block size must be a divisor of input length {inp.shape[0]}')

        mlp_output = mlp_layer(post_attn_norm_output)
        residual += mlp_output

        return residual, present