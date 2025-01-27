# Jacare

River routing in JAX.

## Installation

Install with

```bash
pip install git+https://github.com/limamau/jacare.git
```

To enable JAX in the GPU, it may be necessary to manually install it inside an environment with CUDA and GPUs available through

```bash
pip install -U "jax[cuda12]"
```

## Quick start

Build up the toy dataset with `data/generate_dataset.py` and play with the examples using

- [independent catchments](https://github.com/limamau/jacare/tree/main/examples/catchments)

- [complete river routing](https://github.com/limamau/jacare/tree/main/examples/routing)
