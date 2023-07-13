import flax.linen as nn
import jax
import jax.numpy as jnp


class LRNNCell1(nn.Module):

    size: int

    @nn.compact
    def __call__(self, dummy, x):
        n = x.shape[0]
        h = self.variable("state", "h", jnp.zeros, (n, self.size))

        Whx = nn.Dense(self.size)
        Whh = nn.Dense(self.size, use_bias=False)
        Wyh = nn.Dense(self.size)

        h.value = nn.tanh(Whx(x) + Whh(h.value))
        y = nn.tanh(Wyh(h.value))
        return dummy, y


class LRNNCell2(nn.Module):

    size: int

    @nn.compact
    def __call__(self, x):
        n = x.shape[0]
        h = self.variable("state", "h", jnp.zeros, (n, self.size))

        Whx = nn.Dense(self.size)
        Whh = nn.Dense(self.size, use_bias=False)
        Wyh = nn.Dense(self.size)

        h.value = nn.tanh(Whx(x) + Whh(h.value))
        y = nn.tanh(Wyh(h.value))
        return y


if __name__ == "__main__":
    batch_size, seq_len, in_feat, out_feat = 16, 20, 3, 5
    print(f"batch_size: {batch_size}, seq_len: {seq_len}, in_feat: {in_feat}, out_feat: {out_feat}")
    key_1, key_2, key_3 = jax.random.split(jax.random.PRNGKey(0), 3)

    xs = jax.random.uniform(key_1, (batch_size, seq_len, in_feat))
    print(f"input shape: {xs.shape}")

    # approach 1: use dummy variable (scanned function needs to have signature (carry, input) -> (carry, output))
    # model = LRNNCell1(out_feat)
    # scan = nn.transforms.scan(
    #     model, variable_broadcast="params", variable_carry="state", in_axes=1, out_axes=1, split_rngs={"params": False}
    # )
    # variables = model.init(key_3, xs, xs)
    # (dummy, out_val), state = model.apply(variables, xs, xs, mutable=["state"])
    # print(f"dummy shape: {dummy.shape}")
    # print(f"out_val shape: {out_val.shape}")
    # print(f"state shape: {state['state']['h'].shape}")
    # assert out_val.shape == (batch_size, seq_len, out_feat)

    # approach 2: put apply function in scan
    model = LRNNCell2(out_feat)
    state, params = model.init(key_3, jnp.ones((batch_size, in_feat))).pop("params")

    def scan(carry, x):
        y, carry = model.apply({"params": params, **carry}, x, mutable="state")
        return carry, y

    state, out_val = jax.lax.scan(scan, state, jnp.swapaxes(xs, 0, 1))
    print(f"out_val shape: {out_val.shape}")
    print(f"state shape: {state['state']['h'].shape}")
    assert jnp.swapaxes(out_val, 0, 1).shape == (batch_size, seq_len, out_feat)
