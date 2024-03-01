import jax
import optax
from tqdm import tqdm
import jax.numpy as jnp
from flax.training import train_state
from flax.training import checkpoints
import tensorboardX as tbx

from model import Model
from dataloader import *


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
    tbx_writer: tbx.SummaryWriter

    def train_step(self, state, batch):
        def loss_fn(params, batch):
            x, y = batch
            return optax.softmax_cross_entropy(
                jax.nn.log_softmax(state.apply_fn(params, x)),
                jax.nn.one_hot(y, 10)
            ).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params, batch)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def test_step(self, state, batch):
        x, y = batch
        logits = state.apply_fn(state.params, x)
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        return accuracy


if __name__ == "__main__":
    # name with mnist
    tbx_writer = tbx.SummaryWriter("logs/mnist")

    key = jax.random.PRNGKey(0)
    epochs = 10
    batch_size = 256
    train_ds, test_ds = get_train_batches(batch_size), get_test_batches(batch_size)

    lr_fn = lr_schedule(2e-3, len(train_ds))

    model = Model()
    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(key, jnp.ones((1, 28, 28, 1))),
        tx=optax.adam(lr_fn),
        tbx_writer=tbx_writer,)

    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_ds)
        for batch in pbar:
            state, loss = state.train_step(state, batch)
            lr = lr_fn(state.step)
            # tbx_writer.add_scalar("loss", loss, state.step)
            # tbx_writer.add_scalar("lr", lr, state.step)
            pbar.set_description(f"epoch: {epoch:3d}, loss: {loss:.4f}, lr: {lr:.4f}")

        if epoch % 1 == 0:
            accuracy = jnp.array([])
            for batch in test_ds:
                acc = state.test_step(state, batch)
                accuracy = jnp.append(accuracy, acc)

            acc = accuracy.mean()
            print(f"epoch: {epoch:3d}, accuracy: {acc:.4f}")
            # tbx_writer.add_scalar("accuracy", acc, state.step)

            checkpoints.save_checkpoint(
                ckpt_dir="/Users/haoyu/Documents/Projects/jax_learning/checkpoints",
                target=state,
                step=epoch,
                overwrite=True,
                keep=2,)

    # tbx_writer.close()
