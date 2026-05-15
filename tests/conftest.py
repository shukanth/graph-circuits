import os

# jax-metal (Apple Silicon) does not support int64 array creation, which JAX
# uses internally when seeding RNG keys. Tests don't need GPU acceleration,
# so force CPU here. The training script will handle Metal separately.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
