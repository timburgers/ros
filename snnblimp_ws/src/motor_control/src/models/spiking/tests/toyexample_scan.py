import functools

import flax.linen as nn
import jax


class LRNNCell(nn.Module):
    @functools.partial(
        nn.transforms.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, h, x):
        nh = h.shape[-1]
        Whx = nn.Dense(nh)
        Whh = nn.Dense(nh, use_bias=False)
        Wyh = nn.Dense(nh)

        h = nn.tanh(Whx(x) + Whh(h))
        y = nn.tanh(Wyh(h))
        return h, y


if __name__ == "__main__":
    batch_size, seq_len, in_feat, out_feat = 16, 20, 3, 5
    print(f"batch_size: {batch_size}, seq_len: {seq_len}, in_feat: {in_feat}, out_feat: {out_feat}")
    key_1, key_2, key_3 = jax.random.split(jax.random.PRNGKey(0), 3)

    xs = jax.random.uniform(key_1, (batch_size, seq_len, in_feat))
    print(f"input shape: {xs.shape}")
    init_carry, _ = nn.LSTMCell.initialize_carry(key_2, (batch_size,), out_feat)
    print(f"init_carry shape: {init_carry.shape}")

    model = LRNNCell()
    variables = model.init(key_3, init_carry, xs)
    out_carry, out_val = model.apply(variables, init_carry, xs)
    print(f"out_carry shape: {out_carry.shape}")
    print(f"out_val shape: {out_val.shape}")

    assert out_val.shape == (batch_size, seq_len, out_feat)
