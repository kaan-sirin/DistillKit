#!/bin/bash

# Install PyTorch
uv add torch

# Install wheel, packaging, and ninja
uv add wheel packaging ninja

# Install flash-attn and deepspeed
uv add flash-attn deepspeed

# Install requirements from requirements.txt
uv pip install -r requirements.txt

