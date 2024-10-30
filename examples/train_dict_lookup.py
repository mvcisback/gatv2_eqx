# ruff: noqa: F722,F821

import dataclasses
from functools import partial
from typing import Sequence

import equinox as eqx
import funcy as fn
import jax
import jax.numpy as jnp
import numpy as np
import optax 
from beartype import beartype
from dict_lookup_mpnn_problem import gen_problems
from dict_lookup_mpnn_problem.generate import Problem
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree, jaxtyped

from gatv2_eqx import GATv2


def pad_problem(problem: Problem, target_nodes: int) -> Problem:
    n, channels = problem.nodes.shape
    nodes = np.zeros((target_nodes, channels))
    nodes[:n] = problem.nodes
    adj = np.eye(target_nodes)
    adj[:n,:n] = problem.adj

    # HACK: fake answers are negative.
    answers = -np.ones((target_nodes >> 1), dtype=np.int32)
    answers[:len(problem.answers)] = problem.answers

    return dataclasses.replace(problem, nodes=nodes, adj=adj, answers=answers)


type ProblemBatch = tuple[
    Float[Array, "batch nodes channel"],
    Float[Array, "batch nodes nodes"],
    Int[Array, "batch+nodes"],
]


def merge_batch(problems: Sequence[Problem]) -> ProblemBatch:
    nodes = jnp.stack([p.nodes for p in problems])
    adj = jnp.stack([p.adj for p in problems])
    answers = jnp.stack([p.answers for p in problems])
    return (nodes, adj, answers)


@jaxtyped(typechecker=beartype)
class LookUpNetwork(eqx.Module):
    gnn: GATv2
    decoder: eqx.nn.Linear
    gnn_iters: int = eqx.static_field()

    def __init__(self,
                 n_keys: int,
                 n_vals: int,
                 gnn_iters: int = 1,
                 *,
                 rng_key: PRNGKeyArray):
        self.gnn_iters = gnn_iters
        key_messenger, key_decoder = jax.random.split(rng_key)
        n = n_keys + n_vals
        self.gnn = GATv2(n_features=n, key=key_messenger)
        self.decoder = eqx.nn.Linear(n, n_vals, key=key_decoder) 

    def __call__(self,
                 nodes: Float[Array, "nodes keys+values"],
                 adj:   Float[Array, "nodes nodes"],
                 *,
                 key: PRNGKeyArray) -> Float[Array, "keys values"]:
        """Returns the log belief over the values for each key node."""
        n = nodes.shape[0]
        latent_nodes = self.gnn(nodes, 
                                adj + jnp.eye(n),
                                n_iters=self.gnn_iters, key=key)
        # First half of the nodes only have the keys.
        # Second half of the nodes have keys + values.
        # The goal is to learn to complete the value.
        latent_nodes = latent_nodes[:(n // 2)]
        value_scores = jax.vmap(self.decoder)(latent_nodes)
        return jax.nn.log_softmax(value_scores)

    @staticmethod
    def train(n_keys: int,
              n_vals: int,
              gnn_iters: int = 1,
              epochs: int = 10_000,
              batch_size: int = 200,
              learning_rate: float = 1e-3,
              *,
              jax_seed: int = 0,
              problems_seed: int = 2):

        def loss(nodes: Float[Array, "batch nodes channel"],
                 adj: Float[Array, "batch nodes nodes"],
                 answers: Int[Array, "batch nodes"],
                 key: PRNGKeyArray,
                 *,
                 model: LookUpNetwork) -> float:
            log_beliefs = model(nodes=nodes, adj=adj, key=key)

            # Cross entropy loss, i.e., average surprisal of the answers.
            losses = jax.vmap(lambda a, logp: -logp[a])(answers, log_beliefs)
            losses = jnp.where(answers >= 0, losses, 0.)
            return losses.mean()

        def batch_loss(model: LookUpNetwork,
                       nodes: Float[Array, "batch nodes channel"],
                       adj: Float[Array, "batch nodes nodes"],
                       answers: Int[Array, "batch nodes"],
                       key: PRNGKeyArray) -> float:
            n_batches = nodes.shape[0]
            keys = jax.random.split(key, n_batches)
            losses = jax.vmap(partial(loss, model=model))
            return losses(nodes, adj, answers, keys).mean()

        loss_and_grad = eqx.filter_value_and_grad(batch_loss)

        key = jax.random.PRNGKey(jax_seed)
        init_key, *keys = jax.random.split(key, 1 + epochs)
        model = LookUpNetwork(n_keys=n_keys,
                              n_vals=n_vals,
                              gnn_iters=gnn_iters,
                              rng_key=init_key)

        optim = optax.adam(learning_rate)
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        dataset = gen_problems(n_keys=n_keys, n_vals=n_vals, seed=problems_seed)

        # Create training batches and test set.
        #
        # For jit performance reasons these are padded to have the same
        # number of nodes and stacked together.
        dataset = map(partial(pad_problem, target_nodes=2*n_keys), dataset)
        test = fn.take(100, dataset)
        test = merge_batch(test)
        batches = fn.chunks(batch_size, dataset)
        batches = map(merge_batch, batches)

        @eqx.filter_jit
        def make_step(model: LookUpNetwork,
                      opt_state: PyTree,
                      nodes: Float[Array, "batch nodes channel"],
                      adj: Float[Array, "batch nodes nodes"],
                      answers: Int[Array, "batch nodes"],
                      key: PRNGKeyArray):
            key1, key2 = jax.random.split(key)
            val, grads = loss_and_grad(model, nodes, adj, answers, key=key1)
            updates, opt_state = optim.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            test_loss = batch_loss(model, *test, key=key2)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, val, test_loss


        for i, (epoch_key, batch) in enumerate(zip(keys, batches)):
            print(f"epoch: {i}")
            model, opt_state, val, test_loss = make_step(model, opt_state, *batch, key=epoch_key)
            print(f"loss: {val}, test_loss: {test_loss}")


if __name__ == '__main__':
    model = LookUpNetwork.train(n_keys=3, n_vals=10)
