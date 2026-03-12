"""NavigationGraph — NetworkX-based topological navigation graph.

Represents a campus/building as zones (rooms, corridors) connected by
passages (doors, openings).  Supports hierarchical metadata
(block/floor/room/location) produced by ``oneformer3d2navgraph.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np


class NavigationGraph:
    """Topological navigation graph backed by a NetworkX undirected graph.

    Each **node** is a zone with attributes: name, centroid, area, hierarchy,
    navigable_points_file.

    Each **edge** is a passage with attributes: passage_point, width, distance.
    """

    def __init__(self) -> None:
        self.graph: nx.Graph = nx.Graph()
        self.locations: list[dict[str, Any]] = []
        self.hierarchy: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_zone(
        self,
        zone_id: str,
        name: str,
        centroid: list[float],
        area: float,
        navigable_points_file: str | None = None,
        hierarchy: dict[str, str] | None = None,
    ) -> None:
        self.graph.add_node(
            zone_id,
            name=name,
            centroid=centroid,
            area=area,
            navigable_points_file=navigable_points_file or "",
            hierarchy=hierarchy or {},
        )

    def add_connection(
        self,
        zone_a: str,
        zone_b: str,
        passage_point: list[float],
        width: float,
        door_instance_id: int | None = None,
    ) -> None:
        ca = np.array(self.graph.nodes[zone_a]["centroid"])
        cb = np.array(self.graph.nodes[zone_b]["centroid"])
        distance = float(np.linalg.norm(ca - cb))
        attrs: dict[str, Any] = {
            "passage_point": passage_point,
            "width": width,
            "distance": distance,
        }
        if door_instance_id is not None:
            attrs["door_instance_id"] = door_instance_id
        self.graph.add_edge(zone_a, zone_b, **attrs)

    def add_location(
        self,
        location_id: str,
        name: str,
        room: str,
        position: list[float],
        floor_id: str = "",
        block_id: str = "",
    ) -> None:
        self.locations.append({
            "id": location_id,
            "name": name,
            "room": room,
            "position": position,
            "floor_id": floor_id,
            "block_id": block_id,
        })

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def find_zone_by_name(self, name: str) -> str | None:
        """Return zone_id whose *name* matches (case-insensitive), or None."""
        name_lower = name.lower()
        for nid, data in self.graph.nodes(data=True):
            if data.get("name", "").lower() == name_lower:
                return nid
        return None

    def find_zone_containing_point(self, point: np.ndarray) -> str | None:
        """Return zone_id whose centroid is closest to *point*."""
        best_id: str | None = None
        best_dist = float("inf")
        for nid, data in self.graph.nodes(data=True):
            c = np.array(data["centroid"])
            d = float(np.linalg.norm(c - point))
            if d < best_dist:
                best_dist = d
                best_id = nid
        return best_id

    def find_location_by_name(self, name: str) -> dict[str, Any] | None:
        """Return a location dict matching *name* (case-insensitive)."""
        name_lower = name.lower()
        for loc in self.locations:
            if loc.get("name", "").lower() == name_lower:
                return loc
        return None

    def find_route(self, origin_name: str, dest_name: str) -> list[str]:
        """Return ordered list of zone *names* from origin to destination.

        Names are resolved via ``find_zone_by_name``; if not found they are
        tried as raw zone IDs.  Raises ``nx.NetworkXNoPath`` when no path
        exists and ``KeyError`` when a name cannot be resolved.
        """
        origin_id = self.find_zone_by_name(origin_name) or origin_name
        dest_id = self.find_zone_by_name(dest_name) or dest_name

        if origin_id not in self.graph:
            raise KeyError(f"Zone not found: {origin_name!r}")
        if dest_id not in self.graph:
            raise KeyError(f"Zone not found: {dest_name!r}")

        path_ids = nx.shortest_path(
            self.graph, source=origin_id, target=dest_id, weight="distance"
        )
        return [self.graph.nodes[nid]["name"] for nid in path_ids]

    def find_route_ids(self, origin_id: str, dest_id: str) -> list[str]:
        """Return ordered list of zone *IDs* using Dijkstra on distance."""
        return nx.shortest_path(
            self.graph, source=origin_id, target=dest_id, weight="distance"
        )

    @property
    def zone_ids(self) -> list[str]:
        return list(self.graph.nodes)

    @property
    def zone_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def connection_count(self) -> int:
        return self.graph.number_of_edges()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self, path: str | Path) -> None:
        """Serialize the graph to *navigation_graph.json* format."""
        data = nx.node_link_data(self.graph)
        data["locations"] = self.locations
        data["hierarchy"] = self.hierarchy
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str | Path) -> NavigationGraph:
        """Deserialize from a *navigation_graph.json* file."""
        with open(path) as f:
            data = json.load(f)

        ng = cls()
        ng.locations = data.get("locations", [])
        ng.hierarchy = data.get("hierarchy", [])

        for node in data.get("nodes", []):
            nid = node["id"]
            ng.add_zone(
                zone_id=nid,
                name=node.get("name", nid),
                centroid=node.get("centroid", [0, 0, 0]),
                area=node.get("area", 0.0),
                navigable_points_file=node.get("navigable_points_file"),
                hierarchy=node.get("hierarchy"),
            )

        for edge in data.get("edges", []):
            src = edge.get("source", edge.get("zone_a"))
            tgt = edge.get("target", edge.get("zone_b"))
            ng.graph.add_edge(
                src,
                tgt,
                passage_point=edge.get("passage_point", [0, 0, 0]),
                width=edge.get("width", 0.0),
                distance=edge.get("distance", 0.0),
                **{
                    k: v
                    for k, v in edge.items()
                    if k not in ("source", "target", "zone_a", "zone_b",
                                 "passage_point", "width", "distance")
                },
            )

        return ng

    @classmethod
    def from_navgraph_dir(cls, navgraph_dir: str | Path) -> NavigationGraph:
        """Load from the output directory of ``oneformer3d2navgraph.py``.

        Reads ``navigation_graph.json`` (primary), with ``zones.json`` and
        ``connections.json`` as fallback.
        """
        d = Path(navgraph_dir)
        nav_json = d / "navigation_graph.json"

        if nav_json.is_file():
            ng = cls.from_json(nav_json)
            # Also try loading zone_labels for hierarchy
            labels_path = d / "zone_labels.json"
            if labels_path.is_file() and not ng.hierarchy:
                with open(labels_path) as f:
                    labels = json.load(f)
                ng.hierarchy = labels.get("blocks", [])
            return ng

        # Fallback: build from zones.json + connections.json
        zones_path = d / "zones.json"
        conns_path = d / "connections.json"

        if not zones_path.is_file():
            raise FileNotFoundError(
                f"No navigation_graph.json or zones.json in {d}"
            )

        ng = cls()

        with open(zones_path) as f:
            zones = json.load(f)
        for z in zones:
            ng.add_zone(
                zone_id=z["id"],
                name=z.get("name", z["id"]),
                centroid=z["centroid"],
                area=z.get("area", 0.0),
                navigable_points_file=z.get("navigable_points_file"),
                hierarchy=z.get("hierarchy"),
            )

        if conns_path.is_file():
            with open(conns_path) as f:
                conns = json.load(f)
            for c in conns:
                ng.graph.add_edge(
                    c["zone_a"],
                    c["zone_b"],
                    passage_point=c.get("passage_point", [0, 0, 0]),
                    width=c.get("width", 0.0),
                    distance=c.get("distance", 0.0),
                )

        labels_path = d / "zone_labels.json"
        if labels_path.is_file():
            with open(labels_path) as f:
                labels = json.load(f)
            ng.hierarchy = labels.get("blocks", [])

        return ng

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_passage_points(self, route_ids: list[str]) -> list[list[float]]:
        """Return passage_point coordinates along a zone-ID route."""
        points = []
        for i in range(len(route_ids) - 1):
            edge = self.graph.edges[route_ids[i], route_ids[i + 1]]
            points.append(edge["passage_point"])
        return points

    def __repr__(self) -> str:
        return (
            f"NavigationGraph(zones={self.zone_count}, "
            f"connections={self.connection_count}, "
            f"locations={len(self.locations)})"
        )
