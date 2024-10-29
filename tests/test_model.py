import jax
import jax.numpy as jnp

import equinox as eqx
from dict_lookup_mpnn_problem import gen_problems

from gatv2_eqx import GATv2, GATv2Layer


def test_smoke():
    key = jax.random.PRNGKey(0)
    key_init, key_call = jax.random.split(key)
    model = GATv2(n_features=4, key=key_init)

    adj = jnp.array([[0., 1., 1.],
                     [1., 0., 0.],
                     [0., 1., 0.]])

    # Each node has the unit vector as its feature.
    nodes = jnp.eye(3, 4)
    def f(nodes, key):
        key1, key2 = jax.random.split(key)
        nodes = model(nodes=nodes, adj_mat=adj, n_iters=3, key=key1)
        return nodes.sum()

    f(nodes, key_call)

    df = jax.grad(jax.jit(f))
    grad = df(nodes, key_call)
    assert (grad > 0.0).any()


def test_dict_lookup():
    key = jax.random.PRNGKey(0)
    key_messenger, key_decoder, key_call = jax.random.split(key, 3)

    gnn = GATv2(n_features=3+5, key=key_messenger)
    decoder = eqx.nn.Linear(3+5, 1, key=key_decoder) 
    model = (gnn, decoder)

    def loss(model, *, answers, adj, key, n=3):
        nodes = jnp.array(problem.nodes)
        nodes = gnn(nodes=nodes, adj_mat=adj, n_iters=3, key=key)
        guess = jax.vmap(decoder)(nodes)
        return ((answers - guess)**2).mean()

    problems = gen_problems(n_keys=3, n_vals=5, seed=0)
    problem = next(problems)
    adj = jnp.array(problem.adj)

    loss(model, answers=problem.answers, adj=adj, key=key_call)

    df = jax.jit(jax.grad(loss))
    df(model, answers=problem.answers, adj=adj, key=key_call)
