# ruff: noqa: F722
"""
Implementation of GATv2 from https://arxiv.org/abs/2105.14491.
"""

__all__ = ["GATv2", "GATv2Layer"]  # Public API

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class GATv2Layer(eqx.Module):
    leaky_relu_negative_slope: float = eqx.static_field()

    W_tgt: eqx.nn.Linear  # Project target node.
    W_src: eqx.nn.Linear  # Project src node.
    to_score: eqx.nn.Linear

    def __init__(self,
                 n_features: int,
                 *,
                 leaky_relu_negative_slope: float = 0.2,
                 key: PRNGKeyArray):
        self.leaky_relu_negative_slope = leaky_relu_negative_slope

        key1, key2, key3 = jax.random.split(key, 3)

        # TODO: What is the best way to make these *optionally*
        #       share weights?
        self.W_tgt = eqx.nn.Linear(n_features, n_features, key=key1)
        self.W_src = eqx.nn.Linear(n_features, n_features, key=key2)
        self.to_score = eqx.nn.Linear(n_features, 1, use_bias=False, key=key3)

    def __call__(self,
                 nodes: Float[Array, "node channel"],
                 adj_mat: Float[Array, "node node"],
                 *,
                 key: PRNGKeyArray) -> Float[Array, "node channel"]:
        # Apply linear projects to each node.
        g_src: Float[Array, "node channel"] = jax.vmap(self.W_src)(nodes)
        g_tgt: Float[Array, "node channel"] = jax.vmap(self.W_tgt)(nodes)

        # Create scores of each src given tgt.
        #   g[i, j] = σ(g_src[i] + g_src[j])
        g: Float[Array, "node node channel"]  # tgt x src x channels
        g = jax.vmap(lambda x: x + g_tgt)(g_src)
        g = jax.nn.leaky_relu(g, self.leaky_relu_negative_slope)

        # TODO: Is there a way to avoid flattening and unflattening.
        score: Float[Array, "node node"]
        _score = jax.vmap(self.to_score)(einops.rearrange(g, "tgt src c -> (tgt src) c"))
        score = einops.rearrange(_score, "(tgt src) () -> tgt src", tgt=nodes.shape[0])

        # Each target node normalized across the src dimension.
        mask = jnp.zeros_like(score)
        mask = jnp.where(adj_mat == 0, mask, -float('inf'))
        weights: Float[Array, "node node"]  # tgt -> attention over src.
        weights = jax.nn.softmax(score + mask, axis=1)

        # Average each g_tgt[i] with weights[i].
        return weights @ g_tgt


class GATv2(eqx.Module):
    send_recieve: GATv2Layer
    dropout: eqx.nn.Dropout

    def __init__(self,
                 n_features: int,
                 *,
                 dropout_prob: float = 0.0,
                 leaky_relu_negative_slope: float = 0.2,
                 key: PRNGKeyArray):
        relu_slope = leaky_relu_negative_slope
        self.dropout = eqx.nn.Dropout(dropout_prob)
        self.send_recieve = GATv2Layer(n_features,
                                       leaky_relu_negative_slope=relu_slope,
                                       key=key)

    def __call__(self,
                 nodes: Float[Array, "node channel"],
                 adj_mat: Float[Array, "node node"],
                 n_iters: int,
                 *,
                 key: PRNGKeyArray) -> Float[Array, "node channel"]:
        for key in jax.random.split(key, n_iters):
            nodes = self.dropout(nodes)
            nodes = self.send_recieve(nodes, adj_mat, key=key)
        return nodes
