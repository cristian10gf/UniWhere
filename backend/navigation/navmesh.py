"""NavMesh — Delaunay-based navigation mesh for geometric path planning.

Builds a walkable surface from floor points using 2D Delaunay triangulation,
filters by slope and obstacle clearance, and provides A* pathfinding over
the triangle adjacency graph.
"""

from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay


class NavMesh:
    """Navigation mesh built from 3D floor points.

    Triangulates the XZ projection, lifts back to 3D using original Y
    values, and supports A* path queries on triangle adjacency.
    """

    def __init__(self) -> None:
        self.points: np.ndarray | None = None          # (N, 3)
        self.triangulation: Delaunay | None = None
        self.triangles: np.ndarray | None = None        # (M, 3) vertex indices
        self.centroids: np.ndarray | None = None        # (M, 3)
        self.adjacency: dict[int, list[int]] = {}       # tri_idx → [neighbours]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(
        self,
        floor_points_xyz: np.ndarray,
        slope_threshold: float = 10.0,
        clearance_height: float = 2.0,
        obstacle_points: np.ndarray | None = None,
    ) -> None:
        """Build the NavMesh from 3D floor points.

        Parameters
        ----------
        floor_points_xyz : (N, 3) array of floor surface points.
        slope_threshold : Maximum allowed slope in degrees per triangle.
        clearance_height : Vertical clearance (m) — obstacles above this
            height relative to the floor are ignored.
        obstacle_points : Optional (K, 3) array of obstacle points used
            to cull triangles that overlap obstacles.
        """
        pts = np.asarray(floor_points_xyz, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("floor_points_xyz must be (N, 3)")

        self.points = pts
        xz = pts[:, [0, 2]]

        self.triangulation = Delaunay(xz)
        all_simplices = self.triangulation.simplices  # (M, 3)

        # ---- Filter by slope ----
        keep = np.ones(len(all_simplices), dtype=bool)
        for i, tri in enumerate(all_simplices):
            v0, v1, v2 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-12:
                keep[i] = False
                continue
            normal /= norm_len
            # Slope = angle between normal and vertical (0,1,0)
            cos_angle = abs(normal[1])
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            if angle_deg > slope_threshold:
                keep[i] = False

        # ---- Filter by obstacle clearance ----
        if obstacle_points is not None and len(obstacle_points) > 0:
            obs = np.asarray(obstacle_points, dtype=np.float64)
            # Keep only obstacles within clearance_height of local floor
            for i, tri in enumerate(all_simplices):
                if not keep[i]:
                    continue
                tri_pts = pts[tri]
                centroid_xz = tri_pts[:, [0, 2]].mean(axis=0)
                # Rough radius of triangle in XZ
                radii = np.linalg.norm(
                    tri_pts[:, [0, 2]] - centroid_xz, axis=1
                )
                radius = float(radii.max()) * 1.2
                floor_y = float(tri_pts[:, 1].mean())

                dxz = np.linalg.norm(obs[:, [0, 2]] - centroid_xz, axis=1)
                nearby = obs[(dxz < radius)]
                if len(nearby) == 0:
                    continue
                # Check vertical — obstacle within clearance above floor
                height_above = nearby[:, 1] - floor_y
                blocking = (height_above > 0) & (height_above < clearance_height)
                if blocking.any():
                    keep[i] = False

        self.triangles = all_simplices[keep]

        # ---- Centroids ----
        self.centroids = np.array([
            pts[tri].mean(axis=0) for tri in self.triangles
        ])

        # ---- Build adjacency from shared edges ----
        self._build_adjacency()

    def _build_adjacency(self) -> None:
        """Build triangle adjacency dict from shared edges."""
        edge_to_tri: dict[tuple[int, int], list[int]] = {}
        for ti, tri in enumerate(self.triangles):
            for j in range(3):
                e = (min(tri[j], tri[(j + 1) % 3]),
                     max(tri[j], tri[(j + 1) % 3]))
                edge_to_tri.setdefault(e, []).append(ti)

        self.adjacency = {i: [] for i in range(len(self.triangles))}
        for tris in edge_to_tri.values():
            if len(tris) == 2:
                a, b = tris
                self.adjacency[a].append(b)
                self.adjacency[b].append(a)

    # ------------------------------------------------------------------
    # Pathfinding
    # ------------------------------------------------------------------

    def find_path(
        self, start_xyz: np.ndarray, end_xyz: np.ndarray
    ) -> list[list[float]]:
        """Find a path from *start_xyz* to *end_xyz* as 3D waypoints.

        Uses A* on the triangle adjacency graph with Euclidean heuristic.
        Returns an empty list if no path is found.
        """
        if self.centroids is None or len(self.centroids) == 0:
            return []

        start = np.asarray(start_xyz, dtype=np.float64)
        end = np.asarray(end_xyz, dtype=np.float64)

        start_tri = self._nearest_triangle(start)
        end_tri = self._nearest_triangle(end)

        if start_tri is None or end_tri is None:
            return []

        if start_tri == end_tri:
            return [start.tolist(), end.tolist()]

        # A* search
        open_set: list[tuple[float, int]] = [(0.0, start_tri)]
        came_from: dict[int, int] = {}
        g_score: dict[int, float] = {start_tri: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end_tri:
                break

            for neighbour in self.adjacency.get(current, []):
                tentative = g_score[current] + float(
                    np.linalg.norm(
                        self.centroids[neighbour] - self.centroids[current]
                    )
                )
                if tentative < g_score.get(neighbour, float("inf")):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative
                    h = float(np.linalg.norm(
                        self.centroids[neighbour] - self.centroids[end_tri]
                    ))
                    heapq.heappush(open_set, (tentative + h, neighbour))

        if end_tri not in came_from and start_tri != end_tri:
            return []

        # Reconstruct path
        path_tris = [end_tri]
        while path_tris[-1] != start_tri:
            path_tris.append(came_from[path_tris[-1]])
        path_tris.reverse()

        waypoints = [start.tolist()]
        for ti in path_tris:
            waypoints.append(self.centroids[ti].tolist())
        waypoints.append(end.tolist())
        return waypoints

    def _nearest_triangle(self, point: np.ndarray) -> int | None:
        """Return index of triangle whose centroid is nearest to *point*."""
        if self.centroids is None or len(self.centroids) == 0:
            return None
        dists = np.linalg.norm(self.centroids - point, axis=1)
        return int(np.argmin(dists))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_ply(self, path: str | Path) -> None:
        """Export the NavMesh as a PLY file for visualization."""
        if self.points is None or self.triangles is None:
            raise RuntimeError("NavMesh not built yet — call build() first.")

        verts = self.points
        faces = self.triangles

        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(verts)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            for v in verts:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
            for tri in faces:
                f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

    @property
    def triangle_count(self) -> int:
        return len(self.triangles) if self.triangles is not None else 0

    def __repr__(self) -> str:
        return (
            f"NavMesh(points={len(self.points) if self.points is not None else 0}, "
            f"triangles={self.triangle_count})"
        )
