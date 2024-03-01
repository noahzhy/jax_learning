import timeit

import jax
import jax.numpy as jnp
import jax_dataloader as jdl
import tensorboardX as tbx


def get_data_loader(bs=32):
    x = jnp.ones((100, 28, 28, 1))
    y = jnp.ones((100,))
    data = jdl.ArrayDataset(x, y)
    return jdl.DataLoader(
        dataset=data,
        backend="jax",
        batch_size=bs,
        shuffle=True,)


if __name__ == "__main__":
    train_loader = get_data_loader()
    for batch in train_loader:
        t = timeit.timeit(lambda: batch, number=1)
        print('time: {} ms'.format(t * 1000))
        break
