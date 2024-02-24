# Jax Learning

## Requirements

- Jax: `pip install jax jaxlib`
- Flax: `pip install flax`
- Optax: `pip install optax`

## Build model

```python
import jax
import jax.numpy as jnp
from flax import linen as nn

class Model(nn.Module):
    def setup(self):
        self.conv1 = nn.Conv(features=16, kernel_size=(3, 3))
        self.conv2 = nn.Conv(features=32, kernel_size=(3, 3))
        self.conv3 = nn.Conv(features=64, kernel_size=(3, 3))
        self.dense1 = nn.Dense(features=256)
        self.dense2 = nn.Dense(features=10)

    @nn.compact
    def __call__(self, x):
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = nn.relu(x)
        x = self.conv3(x)
        x = nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))
        x = self.dense1(x)
        x = nn.relu(x)
        x = self.dense2(x)
        return x
```

## Summary Model

```python
key = jax.random.PRNGKey(0)
x = jnp.ones((1, 28, 28, 1))

model = Model()
tab = model.tabulate(key, x)
print(tab)
```

## Data pipeline

```python
import tensorflow_datasets as tfds


def get_train_batches(batch_size=32):
    ds = tfds.load(name='mnist', split='train', as_supervised=True)
    ds = ds.batch(batch_size).prefetch(1)
    return tfds.as_numpy(ds)


def get_test_batches(batch_size=32):
    ds = tfds.load(name='mnist', split='test', as_supervised=True)
    ds = ds.batch(batch_size).prefetch(1)
    return tfds.as_numpy(ds)
```

## Training

```python
class TrainState(train_state.TrainState):
    def train_step(self, state, batch):
        def loss_fn(params, x, y):
            return optax.softmax_cross_entropy(
                state.apply_fn(params, x),
                jax.nn.one_hot(y, 10),).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params, batch[0], batch[1])
        state = state.apply_gradients(grads=grads)
        return state, loss

    def eval(self, state, batch):
        logits = state.apply_fn(state.params, batch[0])
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch[1])
        return accuracy
```

## Save checkpoint and restore

```python
from flax.training import checkpoints

# Save checkpoint
checkpoints.save_checkpoint(
    ckpt_dir='ckpt', # absolute path
    target=state,
    step=epoch,
    overwrite=True,
    keep=2,
)
```

