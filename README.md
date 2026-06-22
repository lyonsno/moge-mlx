# moge-mlx

MLX-native [MoGe-2](https://github.com/microsoft/MoGe) model components for Apple Silicon.

This repo contains the recovered pure-MLX MoGe-2 port that was originally built inside [`pixal3d-mlx`](https://github.com/lyonsno/pixal3d-mlx): a DINOv2-L/14 backbone, ConvStack neck, points head, mask head, metric scale head, weight loader, and focal/shift recovery path.

It is no longer a README-only placeholder.

## What Works

- `MoGeModel`: 326M parameter MLX model matching the MoGe-2 ViT-L architecture.
- `load_moge_weights`: loads the upstream Hugging Face checkpoint into MLX tensors.
- `model.infer(...)`: image tensor to point map, depth, intrinsics, and mask.
- `estimate_camera_params_mlx(...)`: image file to FOV/distance camera parameters.
- Reference PyTorch/MPS camera path remains available as `estimate_camera_params(...)` for comparison.

This surface is focused on depth/points/mask/intrinsics. Do not read this README as a claim that a normal-map head is wired in this standalone package.

## Install

```bash
git clone https://github.com/lyonsno/moge-mlx.git
cd moge-mlx
uv venv .venv --python python3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Download the upstream MoGe-2 ViT-L checkpoint:

```bash
hf download Ruicheng/moge-2-vitl
```

The loader also accepts an explicit checkpoint/cache path:

```python
from moge_mlx import MoGeModel, load_moge_weights

model = MoGeModel()
load_moge_weights(model, "/path/to/Ruicheng/moge-2-vitl")
```

## Usage

```python
from pathlib import Path

from moge_mlx.camera import estimate_camera_params_mlx

params = estimate_camera_params_mlx(Path("image.png"))
print(params["camera_angle_x"], params["distance"])
```

Lower-level model API:

```python
import mlx.core as mx
import numpy as np
from PIL import Image

from moge_mlx import MoGeModel, load_moge_weights

image = Image.open("image.png").convert("RGB")
image_np = np.array(image).astype(np.float32) / 255.0
image_chw = mx.array(image_np.transpose(2, 0, 1))

model = MoGeModel()
load_moge_weights(model)
result = model.infer(image_chw)
mx.eval(result["depth"], result["intrinsics"])
```

## Evidence And Boundaries

The port was recovered from the `pixal3d-mlx` MoGe camera-estimation branch after the standalone repo had drifted as a placeholder. The recovered branch recorded MLX/PyTorch parity work for camera estimation and point-map post-processing; see [`docs/moge-camera-estimation.md`](docs/moge-camera-estimation.md).

Current boundaries:

- This is a source recovery and standalone packaging pass, not a new public benchmark claim.
- The reference normal-map evidence used to debug the WebGPU lane comes from official/local PyTorch MoGe, not this standalone MLX package.
- Upstream model weights remain governed by their own license and access terms.

## Architecture

```text
moge_mlx/
├── model.py      # DINOv2-L backbone, ConvStack heads, inference post-processing
├── weights.py    # PyTorch/safetensors checkpoint loader and key remapping
└── camera.py     # camera FOV/distance helpers and MLX/reference routes
```

Model outline:

- DINOv2-L/14 backbone: 24 transformer blocks, 1024 dim, 16 heads
- Intermediate feature extraction at layers 5/11/17/23
- 4 output projections summed into the encoder feature map
- ConvStack neck with 5 levels and ConvTranspose/Bilinear resamplers
- Points head, mask head, metric scale head
- Focal/shift recovery through `scipy.optimize.least_squares`

## Tests

Fast checks:

```bash
python -m py_compile moge_mlx/*.py tests/*.py
python -m pytest tests/test_camera.py::TestComputeDistanceFromFov tests/test_model.py::TestMoGeMLXComponentSmoke::test_bilinear_resize
```

Weight-loading and forward tests require the MoGe checkpoint in the Hugging Face cache and enough unified memory for the 326M parameter model.

## Credits

- [MoGe](https://github.com/microsoft/MoGe) by Microsoft Research
- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta
- [MLX](https://github.com/ml-explore/mlx) by Apple
- [`pixal3d-mlx`](https://github.com/lyonsno/pixal3d-mlx), where this MLX MoGe port was first integrated

## License

MIT for the porting code in this repository. Upstream model weights are subject to their own licenses.
