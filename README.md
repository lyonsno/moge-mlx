# moge-mlx

MLX-native [MoGe-2](https://github.com/microsoft/MoGe) monocular geometry estimation for Apple Silicon.

Estimates camera intrinsics, depth, normals, and 3D point maps from a single image. No PyTorch, no CUDA — pure MLX on Metal.

**Status: Early development.** Camera intrinsics estimation works via subprocess bridge to PyTorch/MPS. MLX-native model port in progress.

## What is MoGe?

[MoGe](https://github.com/microsoft/MoGe) (Monocular Geometry Estimation) is a ViT-based model from Microsoft Research that estimates 3D geometry from a single image:

- **Camera intrinsics** (focal length / FOV) — used by [pixal3d-mlx](https://github.com/lyonsno/pixal3d-mlx) for projection-aligned 3D conditioning
- **Depth maps** — metric depth per pixel
- **Surface normals** — per-pixel 3D orientation
- **3D point maps** — full xyz point cloud
- **Object masks** — foreground segmentation

The model is a DINOv2 ViT-L backbone with ConvStack heads for each output modality.

## Why MLX?

The upstream MoGe requires PyTorch + CUDA. A [Perceptasia sidecar](https://github.com/lyonsno/perceptasia) runs it on PyTorch/MPS, and [pixal3d-mlx](https://github.com/lyonsno/pixal3d-mlx) calls it via subprocess for camera estimation. But this adds a PyTorch dependency to an otherwise pure-MLX pipeline.

An MLX-native port would:
- Eliminate the PyTorch dependency entirely
- Share the DINOv2 backbone with DINOv3 (already ported in pixal3d-mlx/trellis2mlx)
- Enable tighter integration with MLX 3D generation pipelines
- Run on Apple Silicon without any CUDA or MPS dependency

## Architecture (upstream)

```
MoGe-2 ViT-L (~300M params)
├── DINOv2 ViT-L backbone (patch embed + 24 transformer layers + RoPE)
├── ConvStack neck
├── ConvStack points_head → 3D point map
├── ConvStack mask_head → foreground mask
├── ConvStack normal_head → surface normals
├── MLP scale_head → metric scale
└── Geometry recovery (focal length + shift from point map)
```

## Roadmap

- [x] Subprocess bridge for camera intrinsics (landed in pixal3d-mlx)
- [ ] DINOv2 ViT-L backbone in MLX (reuse DINOv3 port pattern)
- [ ] ConvStack heads in MLX
- [ ] Geometry recovery (focal/shift from point map)
- [ ] Weight conversion (model.pt → safetensors)
- [ ] End-to-end inference: image → intrinsics + depth + normals
- [ ] Integration with pixal3d-mlx (replace subprocess bridge)

## Credits

- [MoGe](https://github.com/microsoft/MoGe) by Microsoft Research — the model
- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta — the backbone
- [pixal3d-mlx](https://github.com/lyonsno/pixal3d-mlx) — consumer of camera intrinsics
- [MLX](https://github.com/ml-explore/mlx) by Apple — the framework

## License

MIT (porting code). Upstream model weights are subject to their own licenses.
