"""
Training loop and checkpointing.

Will contain:
- train_step: single JIT-compiled forward + backward + Optax update
- train: full training loop with W&B logging and periodic checkpoint saves
- AdamW with weight decay ~1.0 (the known-good grokking setting from Power et al.)
"""
