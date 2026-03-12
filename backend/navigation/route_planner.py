"""RoutePlanner — two-level route planning (topological + geometric).

Combines :class:`NavigationGraph` (zone-level Dijkstra) with per-zone
:class:`NavMesh` instances (triangle-level A*) to produce both high-level
zone sequences and fine-grained 3D waypoint lists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from backend.navigation.nav_graph import NavigationGraph
from backend.navigation.navmesh import NavMesh


class RoutePlanner:
    """Wraps a :class:`NavigationGraph` and optional per-zone NavMeshes."""

    def __init__(
        self,
        nav_graph: NavigationGraph,
        zone_navmeshes: dict[str, NavMesh] | None = None,
    ) -> None:
        self.nav_graph = nav_graph
        self.zone_navmeshes: dict[str, NavMesh] = zone_navmeshes or {}

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_navgraph_dir(
        cls,
        navgraph_dir: str | Path,
        *,
        build_navmeshes: bool = False,
        slope_threshold: float = 10.0,
    ) -> RoutePlanner:
        """Load a planner from a ``oneformer3d2navgraph.py`` output dir.

        Parameters
        ----------
        navgraph_dir : Directory containing navigation_graph.json, zones.json,
            and optionally navigable_points/<zone_id>.npy files.
        build_navmeshes : If True, build NavMesh for each zone that has a
            navigable_points .npy file.
        slope_threshold : Passed to :meth:`NavMesh.build`.
        """
        d = Path(navgraph_dir)
        ng = NavigationGraph.from_navgraph_dir(d)

        zone_navmeshes: dict[str, NavMesh] = {}
        if build_navmeshes:
            for zone_id in ng.zone_ids:
                data = ng.graph.nodes[zone_id]
                pts_file = data.get("navigable_points_file", "")
                pts_path = d / pts_file
                if pts_file and pts_path.is_file():
                    pts = np.load(pts_path)
                    if len(pts) >= 3:
                        nm = NavMesh()
                        nm.build(pts, slope_threshold=slope_threshold)
                        zone_navmeshes[zone_id] = nm

        return cls(ng, zone_navmeshes)

    # ------------------------------------------------------------------
    # Topological planning
    # ------------------------------------------------------------------

    def plan_topological(
        self, origin_name: str, destination_name: str
    ) -> dict[str, Any]:
        """Plan a zone-level route by name.

        Returns::

            {
                "zones": ["Entrance", "Corridor A", "Lab 101"],
                "zone_ids": ["zone_0", "zone_1", "zone_2"],
                "passages": [[x, y, z], ...],
                "total_distance": 42.5,
            }
        """
        ng = self.nav_graph

        origin_id = ng.find_zone_by_name(origin_name) or origin_name
        dest_id = ng.find_zone_by_name(destination_name) or destination_name

        if origin_id not in ng.graph:
            raise KeyError(f"Zone not found: {origin_name!r}")
        if dest_id not in ng.graph:
            raise KeyError(f"Zone not found: {destination_name!r}")

        route_ids = ng.find_route_ids(origin_id, dest_id)
        zone_names = [ng.graph.nodes[nid]["name"] for nid in route_ids]
        passages = ng.get_passage_points(route_ids)

        total_dist = 0.0
        for i in range(len(route_ids) - 1):
            edge = ng.graph.edges[route_ids[i], route_ids[i + 1]]
            total_dist += edge.get("distance", 0.0)

        return {
            "zones": zone_names,
            "zone_ids": route_ids,
            "passages": passages,
            "total_distance": round(total_dist, 3),
        }

    # ------------------------------------------------------------------
    # Geometric planning
    # ------------------------------------------------------------------

    def plan_geometric(
        self,
        origin_xyz: list[float] | np.ndarray,
        destination_xyz: list[float] | np.ndarray,
    ) -> dict[str, Any]:
        """Plan a 3D waypoint route between two points.

        1. Determine origin and destination zones from point proximity.
        2. If same zone and a NavMesh exists: A* within that zone.
        3. If different zones: get topological path, then chain NavMesh
           segments through passage points.

        Returns::

            {
                "zones": [...],
                "zone_ids": [...],
                "waypoints": [[x, y, z], ...],
                "total_distance": 42.5,
            }
        """
        ng = self.nav_graph
        origin = np.asarray(origin_xyz, dtype=np.float64)
        dest = np.asarray(destination_xyz, dtype=np.float64)

        origin_zone = ng.find_zone_containing_point(origin)
        dest_zone = ng.find_zone_containing_point(dest)

        if origin_zone is None or dest_zone is None:
            raise ValueError("Could not locate origin/destination in any zone.")

        # Same zone — direct NavMesh path
        if origin_zone == dest_zone:
            waypoints = self._navmesh_path(origin_zone, origin, dest)
            return {
                "zones": [ng.graph.nodes[origin_zone]["name"]],
                "zone_ids": [origin_zone],
                "waypoints": waypoints,
                "total_distance": _path_length(waypoints),
            }

        # Different zones — topological + geometric chaining
        route_ids = ng.find_route_ids(origin_zone, dest_zone)
        zone_names = [ng.graph.nodes[nid]["name"] for nid in route_ids]
        passages = ng.get_passage_points(route_ids)

        waypoints: list[list[float]] = []
        current_point = origin

        for i, zone_id in enumerate(route_ids):
            if i < len(passages):
                # Navigate from current_point to the passage exiting this zone
                target = np.array(passages[i])
            else:
                # Last zone — navigate to destination
                target = dest

            segment = self._navmesh_path(zone_id, current_point, target)
            # Avoid duplicating the junction point
            if waypoints and segment and np.allclose(waypoints[-1], segment[0]):
                segment = segment[1:]
            waypoints.extend(segment)
            current_point = target

        return {
            "zones": zone_names,
            "zone_ids": route_ids,
            "waypoints": waypoints,
            "total_distance": _path_length(waypoints),
        }

    def _navmesh_path(
        self,
        zone_id: str,
        start: np.ndarray,
        end: np.ndarray,
    ) -> list[list[float]]:
        """Get NavMesh A* path within a zone, or fall back to straight line."""
        if zone_id in self.zone_navmeshes:
            nm = self.zone_navmeshes[zone_id]
            path = nm.find_path(start, end)
            if path:
                return path
        # Fallback: direct straight-line segment
        return [start.tolist(), end.tolist()]

    def __repr__(self) -> str:
        return (
            f"RoutePlanner(graph={self.nav_graph}, "
            f"navmeshes={len(self.zone_navmeshes)})"
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _path_length(waypoints: list[list[float]]) -> float:
    """Sum of Euclidean distances between consecutive waypoints."""
    total = 0.0
    for i in range(len(waypoints) - 1):
        a = np.array(waypoints[i])
        b = np.array(waypoints[i + 1])
        total += float(np.linalg.norm(b - a))
    return round(total, 3)
