import os

# Hard assignment, not setdefault: jax-metal may register as default backend
# even without an explicit env var, which setdefault would not override.
os.environ["JAX_PLATFORMS"] = "cpu"
