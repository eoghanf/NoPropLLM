# NoProp Language Modeling

This repository implements the NoProp method for training transformer-based language models using diffusion techniques instead of traditional backpropagation. The method adapts the progressive denoising approach from the NoProp paper to perform next-token prediction on large-scale text datasets.

## Overview

Instead of using backpropagation to train neural networks end-to-end, NoProp trains each transformer layer to denoise progressively less noisy token embeddings. This diffusion-based approach enables independent layer training while maintaining the sequential nature of language modeling.

## Dataset

The project uses the FineWeb dataset, a large-scale web text corpus designed for language model training. The dataset is stored in the `data/fineweb10B` directory across 8 files:
- `fineweb_train_000001` through `fineweb_train_000008`

## Quick Start

### 1. Install Dependencies

```bash
conda create -n NoProp python=3.11 pip
conda activate NoProp
pip install -r requirements.txt
```

### 2. Train Language Model on FineWeb

```bash
# Train transformer with NoProp method
python train_fineweb_v2.py
```

### 3. Test Data Loading

```bash
# Verify dataset can be read properly
python -c "from src.dataloaders import distributed_data_generator; next(distributed_data_generator())"
```

## Implementation Details

### NoProp for Language Modeling
- Each transformer layer learns to denoise progressively less noisy token embeddings
- Progressive noise schedule applied across layers
- Next-token prediction objective maintained throughout training
- Independent layer training with separate optimizers

### Architecture
- Multi-layer transformer architecture
- Each layer implemented as a denoising transformer block
- Tokenization and embedding layers for text processing
- Output projection to vocabulary for next-token prediction

### Training Process
1. Input text tokenized and embedded
2. Each transformer layer trained independently to denoise embeddings
3. Progressive noise reduction across layers
4. Next-token prediction loss computed at output layer
5. Separate optimizers for each component

## Data Loading
The project includes distributed data loading capabilities for the FineWeb dataset, with support for:
- Multiple data files (8 shards)
- GPU-based training (single or multi-GPU)
- Efficient batching and tokenization

## Citation

```bibtex
@article{noprop2024,
  title={NoProp: Training Neural Networks Without Back-propagation or Forward-propagation},
  author={Li, Qinyu and Teh, Yee Whye and Pascanu, Razvan},
  journal={arXiv preprint arXiv:2503.24322},
  year={2024}
}
```

