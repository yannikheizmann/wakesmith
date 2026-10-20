# Installation

## Requirements

| Resource | Trial run | Default run |
|---|---|---|
| GPU | NVIDIA, 12 GB | NVIDIA, 24 GB |
| System memory | 16 GB | 32 GB (VoxCPM2 peaks near 12 GB) |
| Disk | ~35 GB cache + 5 GB per run | ~45 GB cache + 10 GB per run |
| Time on an RTX 4090 | ~30 min | ~4 h |

wakesmith runs on Linux x86-64. The CUDA 12.8 wheels in the lock file support RTX 30xx, 40xx
and 50xx GPUs with a current NVIDIA driver (`nvidia-smi` must work).

## Setup

```bash
git clone https://github.com/yannikheizmann/wakesmith.git
cd wakesmith
uv sync --frozen
```

No system packages are needed: espeak-ng is bundled by `espeakng-loader`, and all models and
datasets are downloaded on the first run.

## Cache location

Datasets and weights (~30 GB) are stored in `~/.cache/wakesmith`, or `$XDG_CACHE_HOME/wakesmith`.
Pass `--cache-dir` to use other storage; slower network storage is fine for the cache, while
the run directory (`--work-dir`) benefits from a local SSD. Set `HF_TOKEN` to avoid Hugging Face
rate limits on the first download.
