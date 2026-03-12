# UniWhere Relocalization

ACE-based 6DoF visual relocalization module. Loads trained ACE models and estimates camera poses from query images using **OpenCV PnP+RANSAC** (no dsacstar/CUDA build required).

## Quick Start

```python
from backend.relocalization import ACERelocalizer

relocalizer = ACERelocalizer(
    encoder_path="preprocesamiento/models/ace/ace_encoder_pretrained.pt",
    head_path="preprocesamiento/data/output/my_scene.pt",
)

import cv2
image = cv2.imread("query.jpg")
result = relocalizer.relocalize(image, focal_length=600.0)

if result.success:
    print(f"Position: {result.translation}")
    print(f"Inliers: {result.inlier_count}")
    print(f"Pose:\n{result.pose}")
```

## Model Files

| File | Location | Description |
|------|----------|-------------|
| Encoder | `preprocesamiento/models/ace/ace_encoder_pretrained.pt` | Scene-agnostic pretrained encoder (shared across all scenes) |
| Head | `preprocesamiento/data/output/<scene>.pt` | Scene-specific head trained by the preprocessing pipeline |

The preprocessing pipeline (`preprocesamiento/pipelines/pipeline.sh --run-ace`) produces head weights at `preprocesamiento/data/output/<scene>.pt`. The encoder is pretrained and ships with the ACE model.

## API Reference

### `ACERelocalizer(encoder_path, head_path, device=None, image_height=480)`

- **encoder_path** — Path to pretrained encoder `.pt` file
- **head_path** — Path to scene-specific head `.pt` file
- **device** — `"cuda"`, `"cpu"`, or `None` (auto-detect)
- **image_height** — Target height for input images (default: 480)

### `ACERelocalizer.relocalize(image, focal_length, principal_point=None)`

- **image** — BGR `(H, W, 3)` or grayscale `(H, W)` uint8 numpy array
- **focal_length** — Camera focal length in pixels at original image resolution
- **principal_point** — `(cx, cy)` in pixels; defaults to image center

Returns a `RelocalizationResult`:

| Field | Type | Description |
|-------|------|-------------|
| `pose` | `np.ndarray (4,4)` | Camera-to-world transformation matrix |
| `translation` | `np.ndarray (3,)` | Camera position in world coordinates |
| `rotation` | `np.ndarray (3,3)` | Camera orientation as rotation matrix |
| `inlier_count` | `int` | Number of RANSAC inliers |
| `success` | `bool` | Whether pose estimation succeeded |

## Integration with Navigation

The `translation` field feeds directly into the navigation module:

```python
from backend.relocalization import ACERelocalizer
from backend.navigation.nav_graph import NavigationGraph
from backend.navigation.route_planner import RoutePlanner

# Relocalize
relocalizer = ACERelocalizer(encoder_path=..., head_path=...)
result = relocalizer.relocalize(image, focal_length=600.0)

# Find current zone
graph = NavigationGraph.from_file("campus_graph.json")
zone = graph.find_zone_containing_point(result.translation)

# Plan route
planner = RoutePlanner(graph)
route = planner.plan_geometric(origin=result.translation, destination=target_point)
```

## Architecture

```
query image (BGR/grayscale)
    │
    ▼
  Preprocessing (grayscale → resize → normalize)
    │
    ▼
  Encoder (FCN, scene-agnostic)  ──→  features (512, H/8, W/8)
    │
    ▼
  Head (MLP, scene-specific)     ──→  scene coordinates (3, H/8, W/8)
    │
    ▼
  PnP+RANSAC (OpenCV)           ──→  4×4 camera-to-world pose
```

- **Encoder**: Pretrained FCN that extracts features from grayscale images. Shared across scenes.
- **Head**: Scene-specific MLP (1×1 convolutions) that predicts 3D scene coordinates per pixel.
- **PnP solver**: OpenCV `solvePnPRansac()` replaces dsacstar for portable inference.

## Dependencies

Managed via UV (`pyproject.toml`):

- `torch` / `torchvision` — neural network inference
- `opencv-python-headless` — PnP+RANSAC pose solver
- `numpy` — array operations
