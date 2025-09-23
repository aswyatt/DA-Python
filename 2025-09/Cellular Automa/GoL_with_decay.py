# from __future__ import annotations

type Cell = tuple[int, int]

# Precomputed neighbor offsets (Moore neighborhood, 8 neighbors)
_NEIGHBOR_OFFSETS: tuple[Cell, ...] = (
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
)

def next_generation(live: set[Cell]) -> set[Cell]:
    """
    Compute one generation of Conway's Life on an unbounded plane.

    live: set of (x, y) integer coordinates representing live cells.
    returns: new set of live cells (fresh set; input is not modified).
    """
    if not live:
        return set()

    # Count neighbors only around live cells and their neighbors.
    counts: dict[Cell, int] = {}
    counts_get = counts.get  # local bindings for speed
    for (x, y) in live:
        for dx, dy in _NEIGHBOR_OFFSETS:
            key = (x + dx, y + dy)
            counts[key] = counts_get(key, 0) + 1

    # Apply B3/S23
    new_live: set[Cell] = set()
    add = new_live.add
    is_live = live.__contains__
    for cell, c in counts.items():
        # Birth on 3, survival on 2 for currently-live cells
        if c == 3 or (c == 2 and is_live(cell)):
            add(cell)

    return new_live