_weighted_sum_values = partial(jnp.einsum, 'hts,hsd->htd')
class LlamaAttention(ConfigModule):
    def __init__(self, pos_emb, use_flash_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = pos_emb
        self.use_flash_attention = use_flash_attention
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past,
        no_cache_update=False
    ):

        # n_head x time x head_dim
        queries, keys, values = [
            checkpoint_name(
                hk.Linear(self.config.hidden_size, name=f'{name}_proj', with_bias=False)(hidden_states).reshape(
                    -1, self.config.num_attention_heads, self.head_dim
                ).transpose(1, 0, 2),
                name=f'llama_{name}_proj'
            )
            for name in 'qkv'
        ]

        queries, keys = self.pos_emb(queries, keys, position_ids)

        q_len = hidden_states.shape[-2]

        if no_cache_update:
            output, new_past = self._attention_with_no_cache_update(queries, keys, values, past, attention_mask, q_len)
        else:
            output, new_past = self._attention_with_cache(queries, keys, values, past, attention_mask, q_len)

        assert output.shape == (self.config.num_attention_heads, q_len, self.head_dim)
        output = output.transpose(1, 0, 2).reshape(-1, self.config.hidden_size)

        result = hk.Linear(self.config.hidden_size, name='o_proj', with_bias=False)(output.reshape(-1, self.config.hidden_size))
        return result, new_past

    def _attention_with_cache(self, queries, keys, values, past, attention_mask, q_len):
        past_kv, past_length = past
        if past_kv is not None:
            keys, values = [
                jax.lax.dynamic_update_slice_in_dim(p_tensor, curr_tensor, past_length, axis=1)
                for p_tensor, curr_tensor in zip(past_kv, [keys, values])
            ]

        # should output batch x heads x seq x dim
        if self.use_flash_attention:
            output = causal_flash_attention(queries, keys, values)
        else:
            attention_weights = self._query_key_dot(queries, keys)
            attention_weights += attention_mask
            out_type = attention_weights.dtype
            attention_weights = jax.nn.softmax(attention_weights.astype(jnp.float32), axis=-1)
            attention_weights = attention_weights.astype(out_type)

            expected_kv_size = q_len if past_kv is None else past_kv[0].shape[1]
            assert attention_weights.shape == (self.config.num_attention_heads, q_len, expected_kv_size), f'{attention_weights.shape} != {(self.config.num_attention_heads, q_len, expected_kv_size)}'

            output = _weighted_sum_values(attention_weights, values)
        return output, (keys, values)


    def _attention_with_no_cache_update(self, queries, keys, values, past, attention_mask, q_len):
        if self.use_flash_attention:
            warnings.warn('Cannot use flash attention when `updatekv_after_dot` is passed, using regular attention')

        past_kv, past_length = past

        past_dots = pv = None
        if past_kv is not None:
            pk, pv = past_kv
            past_dots = self._query_key_dot(queries, pk).astype(jnp.float32)

        out_dtype = queries.dtype

        present_dots = self._query_key_dot(queries, keys)
        current_weights = (present_dots + attention_mask).astype(jnp.float32)

        past_value_mean = None
        if past_dots is not None:
            # past dots is heads x queries x keys
            mask = jnp.where(jnp.arange(past_dots.shape[-1]) < past_length, 0, -jnp.inf)
            assert mask.shape == (past_dots.shape[-1],)
            past_dots += mask

            past_lse = jax.scipy.special.logsumexp(past_dots, axis=-1, keepdims=True)
            current_lse = jax.scipy.special.logsumexp(current_weights, axis=-1, keepdims=True)
            total_lse = jnp.logaddexp(past_lse, current_lse)

            past_weights = jnp.exp(past_dots - total_lse).astype(out_dtype)
            current_weights = jnp.exp(current_weights - total_lse).astype(out_dtype)

            past_value_mean = _weighted_sum_values(past_weights, pv)
            curr_value_mean = _weighted_sum_values(current_weights, values)

            value_mean = jnp.where(past_length == 0, curr_value_mean, curr_value_mean + past_value_mean)
        else:
            current_weights = jax.nn.softmax(current_weights, axis=-1).astype(out_dtype)
            value_mean = _weighted_sum_values(current_weights, values)

        assert value_mean.shape == (self.config.num_attention_heads, q_len, self.head_dim)
        return value_mean, (keys, values)

    def _query_key_dot(self, queries, keys):
        return jnp.einsum('htd,hsd->hts', queries, keys) / jnp.sqrt(self.head_dim)