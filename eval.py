import jax
import optax
from tqdm import tqdm
import jax.numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
from flax.training.train_state import TrainState

from model import Model
from dataloader import get_test_batches


@jax.jit
def eval_model(params, x, y):
    logits = model.apply(params, x)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return accuracy


if __name__ == "__main__":
    device = jax.local_devices()
    print(device)

    key = jax.random.PRNGKey(0)
    x = jnp.ones((5, 28, 28, 1))

    model = Model()
    params = model.init(key, x)
    optimizer = optax.adam(1e-3)

    batch_size = 256
    test_ds = get_test_batches(batch_size)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,)

    # restore the model
    state = checkpoints.restore_checkpoint(
        ckpt_dir="checkpoints",
        target=state,
        step=10,)

    accuracy = jnp.array([])
    pbar = tqdm(test_ds)
    for batch in pbar:
        acc = eval_model(state.params, batch[0], batch[1])
        accuracy = jnp.append(accuracy, acc)

    acc = accuracy.mean()
    print(f"accuracy: {acc:.4f}")