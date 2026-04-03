# RL Snake

A project to use reinforcement learning techniques to let your computer play Snake.

This project is considered as a live experiment, and other algorithms can be integrated in the future to solve the Snake game problem.

## Models

This project currently uses QNet with two affine linear transforms (see `model.py`), using the PyTorch library.

## Setup

You must use Python 3.14.

Using `uv`:

```bash
source .venv/bin/activate # or "uv venv" if no virtual environment is included
uv sync
uv run agent.py
```

## Nix setup

If you run on nixOS, you can try to build the application in a nix shell and run the program in it:

```bash
nix-shell # should load shell.nix
uv run agent.py
```

