"""Informed search algorithms for transfer-path optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from typing import Callable

import networkx as nx


Heuristic = Callable[[str, str, nx.DiGraph], float]


@dataclass
class SearchResult:
    path: list[str]
    total_cost: float
    nodes_expanded: int
    expansion_order: list[str] = field(default_factory=list)


def path_cost(graph: nx.DiGraph, path: list[str]) -> float:
    if len(path) < 2:
        return 0.0
    return sum(float(graph[path[index]][path[index + 1]]["weight"]) for index in range(len(path) - 1))


def reconstruct_path(came_from: dict[str, str], current: str) -> list[str]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return list(reversed(path))


def astar_search(graph: nx.DiGraph, start: str, goal: str, heuristic: Heuristic) -> SearchResult:
    tie_breaker = count()
    open_heap: list[tuple[float, int, str]] = []
    heappush(open_heap, (heuristic(start, goal, graph), next(tie_breaker), start))

    came_from: dict[str, str] = {}
    g_score = {start: 0.0}
    closed: set[str] = set()
    expansion_order: list[str] = []

    while open_heap:
        _, _, current = heappop(open_heap)
        if current in closed:
            continue

        expansion_order.append(current)
        if current == goal:
            path = reconstruct_path(came_from, current)
            return SearchResult(path, path_cost(graph, path), len(expansion_order), expansion_order)

        closed.add(current)

        for neighbor in graph.successors(current):
            tentative_g = g_score[current] + float(graph[current][neighbor]["weight"])
            if tentative_g >= g_score.get(neighbor, float("inf")):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            f_score = tentative_g + heuristic(neighbor, goal, graph)
            heappush(open_heap, (f_score, next(tie_breaker), neighbor))

    return SearchResult([], float("inf"), len(expansion_order), expansion_order)


def greedy_best_first_search(graph: nx.DiGraph, start: str, goal: str, heuristic: Heuristic) -> SearchResult:
    tie_breaker = count()
    open_heap: list[tuple[float, int, str]] = []
    heappush(open_heap, (heuristic(start, goal, graph), next(tie_breaker), start))

    came_from: dict[str, str] = {}
    visited: set[str] = set()
    expansion_order: list[str] = []

    while open_heap:
        _, _, current = heappop(open_heap)
        if current in visited:
            continue

        visited.add(current)
        expansion_order.append(current)
        if current == goal:
            path = reconstruct_path(came_from, current)
            return SearchResult(path, path_cost(graph, path), len(expansion_order), expansion_order)

        for neighbor in graph.successors(current):
            if neighbor in visited:
                continue
            came_from.setdefault(neighbor, current)
            heappush(open_heap, (heuristic(neighbor, goal, graph), next(tie_breaker), neighbor))

    return SearchResult([], float("inf"), len(expansion_order), expansion_order)

