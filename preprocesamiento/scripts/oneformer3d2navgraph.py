#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scipy",
#     "plyfile",
# ]
# ///
"""
Convert a OneFormer3D labeled point cloud into a navigation graph.

Reads a labeled PLY (from infer.py) with semantic_label and instance_id per
point, extracts walkable zones from floor points, detects doors as zone
connections, and builds a topological navigation graph.

Produces:
  - zones.json:            detected zones with centroids and areas
  - connections.json:      edges between zones through doors
  - navigation_graph.json: full graph (NetworkX node-link format)
  - zone_labels.json:      user-editable hierarchy config (blocks/floors/rooms/locations)
  - navigable_points/      per-zone floor point clouds (.npy)

Hierarchy (configured via zone_labels.json):
  block → floor → room → location
  - block:    a building or separate structure
  - floor:    a set of rooms on the same level, connected by corridors
  - room:     an individual detected zone (mapped from auto-detected zone_N)
  - location: a named point of interest within a room (e.g., "vending machine")

Usage:
    uv run oneformer3d2navgraph.py --input labeled.ply --output-dir navgraph/
    uv run oneformer3d2navgraph.py --input labeled.ply --output-dir navgraph/ --labels zone_labels.json
    uv run oneformer3d2navgraph.py --input labeled.ply --output-dir navgraph/ --voxel-size 0.15
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import KDTree

# S3DIS semantic class IDs
S3DIS_CLASSES = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "table",
    8: "chair",
    9: "sofa",
    10: "bookcase",
    11: "board",
    12: "clutter",
}


# ---------------------------------------------------------------------------
# PLY loading
# ---------------------------------------------------------------------------

def load_labeled_ply(ply_path: Path) -> dict:
    """Load a labeled PLY file produced by OneFormer3D infer.py.

    Expected vertex properties: x, y, z, red, green, blue, semantic_label, instance_id.

    Returns dict with numpy arrays: coords, colors, semantic_labels, instance_ids.
    """
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]

    coords = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    colors = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.uint8)
    sem = np.array(v["semantic_label"], dtype=np.int32)
    inst = np.array(v["instance_id"], dtype=np.int32)

    print(f"Loaded {len(coords):,} points from {ply_path}")
    return {
        "coords": coords,
        "colors": colors,
        "semantic_labels": sem,
        "instance_ids": inst,
    }


# ---------------------------------------------------------------------------
# Zone extraction (floor connected components)
# ---------------------------------------------------------------------------

def extract_zones(
    coords: np.ndarray,
    semantic_labels: np.ndarray,
    floor_label: int,
    voxel_size: float,
    min_zone_area: float,
) -> list[dict]:
    """Extract walkable zones from floor points using 2D connected components.

    Projects floor points onto the XZ plane (Y is up in COLMAP/S3DIS),
    discretises into a grid, and runs connected-component labelling.

    Returns a list of zone dicts with: id, centroid, area, floor_point_indices.
    """
    floor_mask = semantic_labels == floor_label
    floor_idx = np.where(floor_mask)[0]

    if len(floor_idx) == 0:
        print("WARNING: no floor points found — cannot extract zones.")
        return []

    floor_pts = coords[floor_idx]
    print(f"Floor points: {len(floor_pts):,}")

    # Project to 2D grid (X, Z) — Y is vertical
    xz = floor_pts[:, [0, 2]]
    grid_min = xz.min(axis=0)
    grid_idx = ((xz - grid_min) / voxel_size).astype(np.int32)

    grid_shape = grid_idx.max(axis=0) + 1
    grid = np.zeros(grid_shape, dtype=bool)
    grid[grid_idx[:, 0], grid_idx[:, 1]] = True

    # Connected components
    labeled_grid, n_components = ndimage.label(grid)
    print(f"Connected components found: {n_components}")

    # Map each floor point to its component
    point_component = labeled_grid[grid_idx[:, 0], grid_idx[:, 1]]

    zones = []
    zone_id = 0
    for comp in range(1, n_components + 1):
        comp_mask = point_component == comp
        comp_floor_idx = floor_idx[comp_mask]
        comp_pts = coords[comp_floor_idx]

        area = comp_mask.sum() * (voxel_size ** 2)
        if area < min_zone_area:
            continue

        centroid = comp_pts.mean(axis=0).tolist()
        zones.append({
            "id": f"zone_{zone_id}",
            "name": f"zone_{zone_id}",
            "centroid": centroid,
            "area": round(float(area), 2),
            "navigable_points_file": f"navigable_points/zone_{zone_id}.npy",
            "_floor_point_indices": comp_floor_idx,
        })
        zone_id += 1

    print(f"Zones after filtering (≥{min_zone_area} m²): {len(zones)}")
    return zones


# ---------------------------------------------------------------------------
# Door / connection detection
# ---------------------------------------------------------------------------

def detect_connections(
    coords: np.ndarray,
    semantic_labels: np.ndarray,
    instance_ids: np.ndarray,
    zones: list[dict],
    door_label: int,
) -> list[dict]:
    """Detect connections between zones through door instances.

    For each door instance, finds the 2 closest zone centroids and creates
    an edge between them.
    """
    door_mask = semantic_labels == door_label
    door_idx = np.where(door_mask)[0]

    if len(door_idx) == 0:
        print("WARNING: no door points found — trying proximity-based connections.")
        return _proximity_connections(zones)

    # Group door points by instance
    door_instances: dict[int, np.ndarray] = {}
    for idx in door_idx:
        inst = int(instance_ids[idx])
        if inst not in door_instances:
            door_instances[inst] = []
        door_instances[inst].append(idx)

    print(f"Door instances found: {len(door_instances)}")

    if len(zones) < 2:
        return []

    zone_centroids = np.array([z["centroid"] for z in zones])
    zone_tree = KDTree(zone_centroids)

    connections = []
    seen_pairs: set[tuple[str, str]] = set()

    for inst_id, pt_indices in door_instances.items():
        door_pts = coords[pt_indices]
        door_centroid = door_pts.mean(axis=0)

        # Bounding box width as a proxy for door width
        bbox_size = door_pts.max(axis=0) - door_pts.min(axis=0)
        width = float(max(bbox_size[0], bbox_size[2]))  # XZ width

        # Find 2 nearest zones
        dists, idxs = zone_tree.query(door_centroid, k=min(2, len(zones)))
        if len(zones) < 2:
            continue

        za = zones[idxs[0]]["id"]
        zb = zones[idxs[1]]["id"]
        pair = (min(za, zb), max(za, zb))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        ca = np.array(zones[idxs[0]]["centroid"])
        cb = np.array(zones[idxs[1]]["centroid"])
        distance = float(np.linalg.norm(ca - cb))

        connections.append({
            "zone_a": za,
            "zone_b": zb,
            "passage_point": door_centroid.tolist(),
            "width": round(width, 3),
            "distance": round(distance, 3),
            "door_instance_id": int(inst_id),
        })

    print(f"Connections through doors: {len(connections)}")
    return connections


def _proximity_connections(zones: list[dict], max_distance: float = 5.0) -> list[dict]:
    """Fallback: connect zones whose centroids are within max_distance."""
    connections = []
    for i, za in enumerate(zones):
        for j, zb in enumerate(zones):
            if j <= i:
                continue
            ca = np.array(za["centroid"])
            cb = np.array(zb["centroid"])
            dist = float(np.linalg.norm(ca - cb))
            if dist <= max_distance:
                midpoint = ((ca + cb) / 2).tolist()
                connections.append({
                    "zone_a": za["id"],
                    "zone_b": zb["id"],
                    "passage_point": midpoint,
                    "width": 0.0,
                    "distance": round(dist, 3),
                    "door_instance_id": None,
                })
    print(f"Proximity-based connections (≤{max_distance} m): {len(connections)}")
    return connections


# ---------------------------------------------------------------------------
# Hierarchy labels (zone_labels.json)
# ---------------------------------------------------------------------------

DEFAULT_HIERARCHY = {
    "$schema": "Zone hierarchy configuration for UniWhere navigation graph.",
    "$doc": (
        "Edit this file to organise detected zones into blocks, floors, rooms, "
        "and locations. Each room's 'zones' array maps to auto-detected zone IDs. "
        "Locations are named points of interest inside a room."
    ),
    "blocks": [
        {
            "id": "block_0",
            "name": "Block 0",
            "floors": [
                {
                    "id": "floor_0",
                    "name": "Floor 0",
                    "rooms": [],  # filled from detected zones
                    "locations": [],
                }
            ],
        }
    ],
}


def generate_default_labels(zones: list[dict]) -> dict:
    """Generate a default zone_labels.json with all zones as rooms in one floor/block."""
    labels = json.loads(json.dumps(DEFAULT_HIERARCHY))  # deep copy

    rooms = []
    for z in zones:
        rooms.append({
            "id": z["id"],
            "name": z["name"],
            "zones": [z["id"]],
            "centroid": z["centroid"],
            "area": z["area"],
        })

    labels["blocks"][0]["floors"][0]["rooms"] = rooms
    labels["blocks"][0]["floors"][0]["locations"] = [
        {
            "$example": True,
            "id": "loc_example",
            "name": "Example Location (delete this)",
            "room": rooms[0]["id"] if rooms else "zone_0",
            "position": rooms[0]["centroid"] if rooms else [0, 0, 0],
        }
    ]
    return labels


def load_labels(labels_path: Path) -> dict:
    """Load and validate a user-provided zone_labels.json."""
    with open(labels_path) as f:
        labels = json.load(f)

    if "blocks" not in labels:
        print(f"Error: zone_labels.json must have a 'blocks' key.")
        sys.exit(1)

    return labels


def apply_labels(zones: list[dict], connections: list[dict], labels: dict) -> dict:
    """Merge hierarchy labels into the navigation graph structure.

    Builds a lookup from zone_id → room/floor/block so that each zone node
    in the graph carries its full hierarchy path.
    """
    zone_map = {z["id"]: z for z in zones}

    # Build hierarchy lookup: zone_id → {room, floor, block}
    hierarchy: dict[str, dict] = {}
    for block in labels.get("blocks", []):
        for floor in block.get("floors", []):
            for room in floor.get("rooms", []):
                for zid in room.get("zones", []):
                    hierarchy[zid] = {
                        "room_id": room["id"],
                        "room_name": room.get("name", room["id"]),
                        "floor_id": floor["id"],
                        "floor_name": floor.get("name", floor["id"]),
                        "block_id": block["id"],
                        "block_name": block.get("name", block["id"]),
                    }

    # Enrich zone nodes
    for z in zones:
        if z["id"] in hierarchy:
            z["hierarchy"] = hierarchy[z["id"]]
        else:
            z["hierarchy"] = {
                "room_id": z["id"],
                "room_name": z["name"],
                "floor_id": "unassigned",
                "floor_name": "Unassigned",
                "block_id": "unassigned",
                "block_name": "Unassigned",
            }

    # Collect locations from labels
    locations = []
    for block in labels.get("blocks", []):
        for floor in block.get("floors", []):
            for loc in floor.get("locations", []):
                if loc.get("$example"):
                    continue
                locations.append({
                    "id": loc["id"],
                    "name": loc.get("name", loc["id"]),
                    "room": loc.get("room", ""),
                    "position": loc.get("position", [0, 0, 0]),
                    "floor_id": floor["id"],
                    "block_id": block["id"],
                })

    # Build full graph structure
    graph = {
        "directed": False,
        "multigraph": False,
        "graph": {
            "description": "UniWhere navigation graph",
            "hierarchy_levels": ["block", "floor", "room", "location"],
        },
        "nodes": [],
        "edges": [],
        "locations": locations,
        "hierarchy": labels.get("blocks", []),
    }

    for z in zones:
        node = {
            "id": z["id"],
            "name": z.get("hierarchy", {}).get("room_name", z["name"]),
            "centroid": z["centroid"],
            "area": z["area"],
            "navigable_points_file": z["navigable_points_file"],
            "hierarchy": z.get("hierarchy", {}),
        }
        graph["nodes"].append(node)

    for c in connections:
        edge = {
            "source": c["zone_a"],
            "target": c["zone_b"],
            "passage_point": c["passage_point"],
            "width": c["width"],
            "distance": c["distance"],
        }
        if c.get("door_instance_id") is not None:
            edge["door_instance_id"] = c["door_instance_id"]
        graph["edges"].append(edge)

    return graph


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_outputs(
    output_dir: Path,
    zones: list[dict],
    connections: list[dict],
    graph: dict,
    labels: dict,
    coords: np.ndarray,
    labels_provided: bool,
):
    """Write all output files to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    nav_pts_dir = output_dir / "navigable_points"
    nav_pts_dir.mkdir(parents=True, exist_ok=True)

    # Per-zone navigable points
    for z in zones:
        idx = z.pop("_floor_point_indices")
        pts = coords[idx].astype(np.float32)
        np.save(nav_pts_dir / f"{z['id']}.npy", pts)

    # zones.json (without internal fields)
    zones_clean = []
    for z in zones:
        zones_clean.append({
            "id": z["id"],
            "name": z.get("hierarchy", {}).get("room_name", z["name"]),
            "centroid": z["centroid"],
            "area": z["area"],
            "navigable_points_file": z["navigable_points_file"],
            "hierarchy": z.get("hierarchy", {}),
        })
    _write_json(output_dir / "zones.json", zones_clean)

    # connections.json
    _write_json(output_dir / "connections.json", connections)

    # navigation_graph.json
    _write_json(output_dir / "navigation_graph.json", graph)

    # zone_labels.json — only write if not provided by user (to avoid overwriting edits)
    labels_path = output_dir / "zone_labels.json"
    if not labels_provided:
        _write_json(labels_path, labels)
        print(f"  Generated : {labels_path}")
        print(f"    → Edit this file to assign room/floor/block names, then re-run.")
    else:
        print(f"  Labels    : using provided file (not overwritten)")


def _write_json(path: Path, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Wrote     : {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert OneFormer3D labeled point cloud to a navigation graph "
            "with hierarchical zone labels (block/floor/room/location)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to labeled .ply from OneFormer3D (x,y,z,r,g,b,semantic_label,instance_id).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for navigation graph files.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help=(
            "Path to zone_labels.json with hierarchy config. "
            "If omitted, a default template is generated in output-dir."
        ),
    )
    parser.add_argument(
        "--floor-label",
        type=int,
        default=1,
        help="S3DIS semantic class ID for floor.",
    )
    parser.add_argument(
        "--door-label",
        type=int,
        default=6,
        help="S3DIS semantic class ID for door.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.1,
        help="Grid resolution (m) for zone 2D projection.",
    )
    parser.add_argument(
        "--min-zone-area",
        type=float,
        default=2.0,
        help="Minimum zone area (m²) to keep.",
    )
    parser.add_argument(
        "--proximity-distance",
        type=float,
        default=5.0,
        help="Max distance (m) for proximity-based connections when no doors are found.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_path.is_file():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    # Load labeled point cloud
    data = load_labeled_ply(input_path)

    # Extract zones from floor points
    print("\n── Zone extraction ──")
    zones = extract_zones(
        data["coords"],
        data["semantic_labels"],
        floor_label=args.floor_label,
        voxel_size=args.voxel_size,
        min_zone_area=args.min_zone_area,
    )

    if not zones:
        print("Error: no zones could be extracted. Check that floor points exist.")
        sys.exit(1)

    # Detect connections
    print("\n── Connection detection ──")
    connections = detect_connections(
        data["coords"],
        data["semantic_labels"],
        data["instance_ids"],
        zones,
        door_label=args.door_label,
    )

    # Load or generate hierarchy labels
    print("\n── Hierarchy labels ──")
    labels_provided = args.labels is not None
    if labels_provided:
        labels_path = Path(args.labels).resolve()
        if not labels_path.is_file():
            print(f"Error: labels file not found: {labels_path}")
            sys.exit(1)
        labels = load_labels(labels_path)
        print(f"Loaded labels from: {labels_path}")
    else:
        labels = generate_default_labels(zones)
        print("Generated default labels (edit zone_labels.json to customise).")

    # Build navigation graph with hierarchy
    graph = apply_labels(zones, connections, labels)

    # Save outputs
    print("\n── Output ──")
    save_outputs(
        output_dir, zones, connections, graph, labels,
        data["coords"], labels_provided,
    )

    # Summary
    n_locations = len(graph.get("locations", []))
    n_blocks = len(labels.get("blocks", []))
    n_floors = sum(len(b.get("floors", [])) for b in labels.get("blocks", []))

    print(f"\nNavigation graph complete:")
    print(f"  Zones       : {len(zones)}")
    print(f"  Connections : {len(connections)}")
    print(f"  Blocks      : {n_blocks}")
    print(f"  Floors      : {n_floors}")
    print(f"  Locations   : {n_locations}")
    print(f"  Output      : {output_dir}")


if __name__ == "__main__":
    main()
