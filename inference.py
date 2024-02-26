import glob, random

import jax
import optax
from PIL import Image
import jax.numpy as jnp
from flax.training import checkpoints
from flax.training.train_state import TrainState

from model import Model


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    input_shape = (1, 28, 28, 1)

    model = Model()

    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(key, jnp.ones(input_shape)),
        tx=optax.adam(1e-3),)

    # restore the model
    state = checkpoints.restore_checkpoint(
        ckpt_dir="checkpoints",
        target=state,
        step=10,)

    img_path = random.choice(glob.glob("data/*.jpg"))
    print(f"image path: {img_path}")
    img = Image.open(img_path).convert("L").resize((28, 28))
    img = jnp.array(img).reshape(input_shape)

    logits = model.apply(state.params, img)
    pred = jnp.argmax(logits, axis=-1)
    print(f"prediction: {pred[0]}")
