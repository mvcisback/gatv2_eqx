import dataclasses
from typing import Sequence

import einops
import equinox as eqx
import funcy as fn
import jax
import jax.numpy as jnp
import optax 
from beartype import beartype
from dict_lookup_mpnn_problem import gen_problems
from dict_lookup_mpnn_problem.generate import Problem
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree, jaxtyped

from gatv2_eqx import GATv2


jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# Make Problem a Pytree and thus jit-able.
jax.tree_util.register_pytree_node(
    nodetype=Problem,
    flatten_func=lambda p: (dataclasses.astuple(p), ()),
    unflatten_func=lambda _, args: Problem(*args)
)


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

        def loss(problem: Problem, *,
                 model: LookUpNetwork,
                 key: PRNGKeyArray) -> float:
            answers = jnp.array(problem.answers)
            log_beliefs = model(nodes=jnp.array(problem.nodes),
                                adj=jnp.array(problem.adj),
                                key=key)
            # Cross entropy loss, i.e., average surprisal of the answers.
            return jax.vmap(lambda a, logp: -logp[a])(answers, log_beliefs) \
                      .mean()


        def batch_loss(model: LookUpNetwork,
                       problems: Sequence[Problem],
                       key: PRNGKeyArray) -> float:
            # TODO: Implement padding via isolated nodes
            #       in order to vmap.
            keys = jax.random.split(key, len(problems))
            tmp = 0
            for p, key in zip(problems, keys):
               tmp += loss(p, model=model, key=key)
            return tmp / len(problems)

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
        test = fn.take(100, dataset)
        batches = fn.chunks(batch_size, dataset)
        batch = next(batches)

        @eqx.filter_jit
        def make_step(model: LookUpNetwork,
                      opt_state: PyTree,
                      batch: Sequence[Problem],
                      key: PRNGKeyArray):
            key1, key2 = jax.random.split(key)
            val, grads = loss_and_grad(model, batch, key1)
            updates, opt_state = optim.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            test_loss = batch_loss(model, test, key2)
            model = eqx.apply_updates(model, updates)
            # TODO
            return model, opt_state, val, test_loss


        for i, (epoch_key, _) in enumerate(zip(keys, batches)):
            print(f"epoch: {i}")
            model, opt_state, val, test_loss = make_step(model, opt_state, batch, epoch_key)
            print(f"loss: {val}, test_loss: {test_loss}")


if __name__ == '__main__':
    model = LookUpNetwork.train(n_keys=3, n_vals=10)
