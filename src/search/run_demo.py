"""Run A* versus Greedy Best-First Search on the demo transfer graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from src.search.algorithms import astar_search, greedy_best_first_search
from src.search.transfer_graph import build_demo_transfer_graph, transfer_heuristic


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "search_results.json"
DEFAULT_FIGURE = PROJECT_ROOT / "outputs" / "figures" / "transfer_graph_paths.png"


def draw_paths(graph: nx.DiGraph, astar_path: list[str], greedy_path: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(graph, pos, node_size=1800, node_color="#e5e7eb", edgecolors="#111827")
    nx.draw_networkx_labels(graph, pos, font_size=8)
    nx.draw_networkx_edges(graph, pos, edge_color="#9ca3af", arrows=True, arrowsize=18)
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels={(u, v): data["weight"] for u, v, data in graph.edges(data=True)},
        font_size=8,
    )

    astar_edges = list(zip(astar_path, astar_path[1:]))
    greedy_edges = list(zip(greedy_path, greedy_path[1:]))
    nx.draw_networkx_edges(graph, pos, edgelist=greedy_edges, edge_color="#f59e0b", width=3, arrows=True, arrowsize=20)
    nx.draw_networkx_edges(graph, pos, edgelist=astar_edges, edge_color="#2563eb", width=4, arrows=True, arrowsize=22)
    plt.title("Transfer Graph: A* Path (Blue) vs Greedy Path (Amber)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="Karachi United")
    parser.add_argument("--goal", default="Real Madrid")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()

    graph = build_demo_transfer_graph()
    astar = astar_search(graph, args.start, args.goal, transfer_heuristic)
    greedy = greedy_best_first_search(graph, args.start, args.goal, transfer_heuristic)

    results = {
        "astar": astar.__dict__,
        "greedy_best_first": greedy.__dict__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    draw_paths(graph, astar.path, greedy.path, args.figure)

    print(json.dumps(results, indent=2))
    print(f"Saved graph comparison to {args.figure}")


if __name__ == "__main__":
    main()

