
import jax 
import jax.numpy as jnp

from models.vit import VisionTransformer
from models.resnet import ResNet18


rng = jax.random.PRNGKey(0)
param_rng, dropout_rng = jax.random.split(rng)

# model = ResNet18(small_images=True, use_classifier=False, num_classes=10)
model = VisionTransformer(10, 4, 4, 4, 512, 1024, 0.1, False)

params = model.init({'params': param_rng, 'dropout': dropout_rng}, jnp.ones((1, 32, 32, 3)))

x = jax.random.normal(rng, shape=(1, 32, 32, 3), dtype=jnp.float32)
out = model.apply(params, x, train=True, rngs={'dropout': dropout_rng})['output']

print(out.shape)