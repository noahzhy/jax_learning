import jax
import optax
import pickle
from tqdm import tqdm
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from flax.training import checkpoints

from model import Model
from dataloader import get_train_batches, get_test_batches


class TrainState(train_state.TrainState):
    def train_step(self, state, batch):
        def loss_fn(params, batch):
            x, y = batch
            loss = optax.softmax_cross_entropy(
                state.apply_fn(params, x),
                jax.nn.one_hot(y, 10),
            ).mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params, batch)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def eval(self, state, batch):
        logits = state.apply_fn(state.params, batch[0])
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch[1])
        return accuracy


if __name__ == "__main__":
    device = jax.local_devices()
    print(device)

    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 28, 28, 1))

    model = Model()
    params = model.init(key, x)
    optimizer = optax.adam(1e-3)

    batch_size = 256
    train_ds, test_ds = get_train_batches(batch_size), get_test_batches(batch_size)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,)

    for epoch in range(1, 10 + 1):
        pbar = tqdm(train_ds)
        for batch in pbar:
            state, loss = state.train_step(state, batch)
            pbar.set_description(f"epoch: {epoch:3d}, loss: {loss:.4f}")

        if epoch % 1 == 0:
            accuracy = jnp.array([])
            for batch in test_ds:
                acc = state.eval(state, batch)
                accuracy = jnp.append(accuracy, acc)

            acc = accuracy.mean()
            print(f"epoch: {epoch:3d}, accuracy: {acc:.4f}")

            checkpoints.save_checkpoint(
                ckpt_dir="/Users/haoyu/Documents/Projects/jax_learning/checkpoints",
                target=state,
                step=epoch,
                overwrite=True,
                keep=2,)
