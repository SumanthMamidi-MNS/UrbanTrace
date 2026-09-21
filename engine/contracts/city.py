"""City road-graph and camera-network contracts."""

import networkx as nx
from pydantic import BaseModel


class RoadNode(BaseModel):
    node_id: str
    lat: float
    lon: float


class RoadEdge(BaseModel):
    from_node: str
    to_node: str
    length_m: float
    speed_limit_kmh: float


class Camera(BaseModel):
    camera_id: str
    name: str
    lat: float
    lon: float
    node_id: str
    bearing_deg: float
    is_border: bool


class CityConfig(BaseModel):
    nodes: list[RoadNode]
    edges: list[RoadEdge]
    cameras: list[Camera]

    def to_digraph(self) -> nx.DiGraph:
        """Build a networkx DiGraph from the road nodes/edges.

        Edge weight is travel time in seconds at the speed limit; edges also
        carry length_m and speed_limit_kmh attributes.
        """
        g = nx.DiGraph()
        for node in self.nodes:
            g.add_node(node.node_id, lat=node.lat, lon=node.lon)
        for edge in self.edges:
            travel_time_s = edge.length_m / (edge.speed_limit_kmh * 1000.0 / 3600.0)
            g.add_edge(
                edge.from_node,
                edge.to_node,
                length_m=edge.length_m,
                speed_limit_kmh=edge.speed_limit_kmh,
                weight=travel_time_s,
            )
        return g
