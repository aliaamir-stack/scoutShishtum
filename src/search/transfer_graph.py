"""Demo football transfer graph and admissible heuristic."""

from __future__ import annotations

import networkx as nx


def build_demo_transfer_graph() -> nx.DiGraph:
    """Create a small directed weighted graph for the project demo."""
    graph = nx.DiGraph()
    clubs = {
        "Karachi United": {"tier": 5, "prestige": 20, "budget": 0.3, "competition": 20},
        "Mumbai City": {"tier": 4, "prestige": 35, "budget": 1.5, "competition": 35},
        "Al Hilal": {"tier": 3, "prestige": 55, "budget": 20, "competition": 60},
        "Club Brugge": {"tier": 2, "prestige": 70, "budget": 35, "competition": 65},
        "Ajax": {"tier": 2, "prestige": 78, "budget": 45, "competition": 70},
        "Brighton": {"tier": 1, "prestige": 82, "budget": 65, "competition": 75},
        "Borussia Dortmund": {"tier": 1, "prestige": 88, "budget": 95, "competition": 82},
        "Manchester City": {"tier": 1, "prestige": 96, "budget": 180, "competition": 95},
        "Real Madrid": {"tier": 1, "prestige": 98, "budget": 190, "competition": 98},
    }
    for club, attributes in clubs.items():
        graph.add_node(club, **attributes)

    edges = [
        ("Karachi United", "Mumbai City", 8.0),
        ("Karachi United", "Al Hilal", 19.0),
        ("Mumbai City", "Al Hilal", 9.0),
        ("Mumbai City", "Club Brugge", 16.0),
        ("Al Hilal", "Club Brugge", 11.0),
        ("Al Hilal", "Ajax", 13.0),
        ("Club Brugge", "Brighton", 10.0),
        ("Club Brugge", "Borussia Dortmund", 18.0),
        ("Ajax", "Brighton", 9.0),
        ("Ajax", "Borussia Dortmund", 12.0),
        ("Brighton", "Manchester City", 14.0),
        ("Brighton", "Real Madrid", 17.0),
        ("Borussia Dortmund", "Manchester City", 10.0),
        ("Borussia Dortmund", "Real Madrid", 11.0),
        ("Manchester City", "Real Madrid", 8.0),
    ]
    for source, target, weight in edges:
        graph.add_edge(source, target, weight=weight)

    return graph


def transfer_heuristic(current: str, goal: str, graph: nx.DiGraph) -> float:
    """Composite heuristic using conservative lower-bound components."""
    current_data = graph.nodes[current]
    goal_data = graph.nodes[goal]

    tier_gap = max(0, current_data["tier"] - goal_data["tier"])
    prestige_gap = max(0, goal_data["prestige"] - current_data["prestige"]) / 20
    competition_gap = max(0, goal_data["competition"] - current_data["competition"]) / 40

    return 2.0 * tier_gap + prestige_gap + competition_gap

