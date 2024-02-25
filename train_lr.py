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


def lr_schedule(base_lr, steps_per_epoch, epochs=10, warnup_epochs=2):
    warnup_fn = optax.linear_schedule(
        init_value=0,
        end_value=base_lr,
        transition_steps=steps_per_epoch * warnup_epochs,
    )
    decay_fn = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=steps_per_epoch * (epochs - warnup_epochs),
    )
    schedule_fn = optax.join_schedules(
        schedules=[warnup_fn, decay_fn],
        boundaries=[steps_per_epoch * warnup_epochs],
    )
    return schedule_fn


class TrainState(train_state.TrainState):

    def train_step(self, state, batch, lr_fn):
        def loss_fn(params, batch):
            x, y = batch
            loss = optax.softmax_cross_entropy(
                state.apply_fn(params, x),
                jax.nn.one_hot(y, 10),
            ).mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params, batch)
        state = state.apply_gradients(grads=grads)
        lr = lr_fn(state.step)
        return state, loss, lr

    def eval(self, state, batch):
        logits = state.apply_fn(state.params, batch[0])
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch[1])
        return accuracy


if __name__ == "__main__":
    device = jax.local_devices()
    print(device)

    epochs = 10
    batch_size = 256
    train_ds, test_ds = get_train_batches(batch_size), get_test_batches(batch_size)

    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 28, 28, 1))

    model = Model()
    params = model.init(key, x)
    lr_fn = lr_schedule(2e-3, len(train_ds))
    opt = optax.adam(lr_fn)
    opt_state = opt.init(params)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=opt,)

    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_ds)
        for batch in pbar:
            steps_per_epoch = len(train_ds)
            state, loss, lr = state.train_step(state, batch, lr_fn)
            pbar.set_description(f"epoch: {epoch:3d}, lr: {lr:.6f}, loss: {loss:.4f}")

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