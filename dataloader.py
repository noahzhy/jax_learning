import jax
import jax.numpy as jnp

# load the mnist dataset
import tensorflow_datasets as tfds


def get_train_batches(batch_size=32):
    ds = tfds.load(name='mnist', split='train', as_supervised=True, shuffle_files=True)
    ds = ds.batch(batch_size).prefetch(1)
    return tfds.as_numpy(ds)


def get_test_batches(batch_size=32):
    ds = tfds.load(name='mnist', split='test', as_supervised=True, shuffle_files=False)
    ds = ds.batch(batch_size).prefetch(1)
    return tfds.as_numpy(ds)


if __name__ == "__main__":
    train_ds = get_train_batches()
    test_ds = get_test_batches()
    for batch in train_ds:
        print(batch[0].shape, batch[1].shape)
        break
    for batch in test_ds:
        print(batch[0].shape, batch[1].shape)
        break
