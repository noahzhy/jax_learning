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
def eval_model(params, batch):
    x, y = batch
    logits = model.apply(params, x)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return acc


if __name__ == "__main__":
    device = jax.local_devices()
    print(device)

    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 28, 28, 1))

    model = Model()
    params = model.init(key, x)
    opt = optax.adam(1e-3)

    batch_size = 256
    test_ds = get_test_batches(batch_size)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=opt,)

    # restore the model
    state = checkpoints.restore_checkpoint(
        ckpt_dir="checkpoints",
        target=state,
        step=10,)

    accuracy = jnp.array([])
    for batch in tqdm(test_ds):
        acc = eval_model(state.params, batch)
        accuracy = jnp.append(accuracy, acc)

    acc = accuracy.mean()
    print(f"accuracy: {acc:.4f}")

    import glob, random
    from PIL import Image
    import matplotlib.pyplot as plt

    img_path = random.choice(glob.glob("data/*.jpg"))
    print(f"image path: {img_path}")
    img = Image.open(img_path).convert("L").resize((28, 28))
    img = jnp.array(img).reshape(1, 28, 28, 1)

    # plt.imshow(img[0, :, :, 0], cmap="gray")
    # plt.show()

    logits = model.apply(state.params, img)
    pred = jnp.argmax(logits, axis=-1)
    print(f"prediction: {pred[0]}")
