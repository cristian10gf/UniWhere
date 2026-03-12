# UniWhere Navigation Module

Topological and geometric route planning over 3D-segmented campus models.

## Setup

```bash
uv sync          # install dependencies from pyproject.toml
```

## Data Flow

```
COLMAP dense point cloud (fused.ply)
  → colmap2oneformer3d.py → scene.npy (S3DIS format)
  → OneFormer3D inference  → labeled.ply (semantic + instance labels)
  → oneformer3d2navgraph.py → navigation_graph.json, zones.json, connections.json
  → NavigationGraph / RoutePlanner (this module)
```

## Components

| File | Class | Purpose |
|---|---|---|
| `nav_graph.py` | `NavigationGraph` | NetworkX graph of zones and connections. Load from JSON, query by name, find shortest path (Dijkstra). |
| `navmesh.py` | `NavMesh` | Delaunay triangulation of floor points with slope/obstacle filtering. A\* pathfinding on triangle adjacency. |
| `route_planner.py` | `RoutePlanner` | Two-level planner combining topological (zone-to-zone) and geometric (NavMesh waypoints) routing. |

## Usage

### Topological routing (zone-level)

```python
from backend.navigation import NavigationGraph

g = NavigationGraph.from_navgraph_dir("preprocesamiento/data/scene/navgraph/")
route = g.find_route("entrada", "laboratorio-a")
# → ["entrada", "pasillo-central", "laboratorio-a"]
```

### Full geometric routing (3D waypoints)

```python
from backend.navigation import RoutePlanner

planner = RoutePlanner.from_navgraph_dir(
    "preprocesamiento/data/scene/navgraph/",
    build_navmeshes=True,
)

result = planner.plan_topological("entrada", "laboratorio-a")
# → {"zones": [...], "passages": [...], "total_distance": 42.5}

result = planner.plan_geometric([10.0, 0.0, 5.0], [30.0, 0.0, 20.0])
# → {"zones": [...], "waypoints": [[x,y,z], ...], "total_distance": 55.2}
```

### NavMesh export for visualization

```python
from backend.navigation import NavMesh
import numpy as np

pts = np.load("preprocesamiento/data/scene/navgraph/navigable_points/zone_0.npy")
nm = NavMesh()
nm.build(pts, slope_threshold=10.0)
nm.to_ply("/tmp/navmesh_zone0.ply")
```

## Hierarchy

The navigation graph supports a four-level hierarchy managed via `zone_labels.json`:

- **block** — a building or separate structure
- **floor** — a set of rooms on the same level
- **room** — an individual detected zone (auto-named `zone_N`, user-renamable)
- **location** — a named point of interest within a room (e.g., "vending machine")
