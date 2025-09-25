import sys


def draw_grid(population, N):
    """Draw the grid of size N×N with the current population.
    Show 'O' for live cells, '.' for dead cells."""
    for y in range(N):
        row = ""
        for x in range(N):
            if (x, y) in population:
                row += "O"
            else:
                row += "."
        print(row)
    print()


def step(population, L0):
    """Compute the next generation with life values."""
    new_population = {}

    # First decrement life of all current cells
    decremented = {}
    for (x, y), life in population.items():
        print(life)
        decremented[(x, y)] = life - 1

    # Candidates: all live cells and their neighbours
    candidates = set()
    for x, y in decremented:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                candidates.add((x + dx, y + dy))

    for x, y in candidates:
        # Count live neighbours (after decrement, only those with life >= 0 survive)
        neighbours = sum(
            ((nx, ny) in decremented and decremented[(nx, ny)] >= 0)
            for nx in [x - 1, x, x + 1]
            for ny in [y - 1, y, y + 1]
            if not (nx == x and ny == y)
        )

        if (x, y) in decremented and decremented[(x, y)] >= 0:
            # Surviving cell
            if neighbours in (2, 3):
                new_population[(x, y)] = decremented[(x, y)]
        else:
            # Dead cell may be born
            if neighbours == 3:
                new_population[(x, y)] = L0
    return new_population


def main(N, L0, initial_population):
    """Run the Game of Life with grid size N, initial life L0, and initial population."""
    # population is a dict: (x,y) -> life
    population = {cell: L0 for cell in initial_population}

    while True:
        draw_grid(population, N)
        try:
            input("Press Enter for next step (Ctrl+C to quit)...")
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)
        population = step(population, L0)


if __name__ == "__main__":
    # Example: a glider in a 10x10 grid with L0=4
    N = 25
    L0 = 4
    initial_population = {
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 2),
        (1, 3),
        (2, 3),
        (3, 3),
    }
    initial_population = {(x + 10, y + 10) for x, y in initial_population}
    main(N, L0, initial_population)
