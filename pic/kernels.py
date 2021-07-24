from definitions import *

def initialize_grid(tile: BoundingBox):
    n_columns = tile.right-tile.left+1
    n_rows = tile.top-tile.bottom+1
    grid = np.ndarray((n_columns, n_rows), dtype=np.float64)

    for y in range(tile.bottom, tile.top+1):
        for x in range(tile.left, tile.right+1):
            grid[y-tile.bottom, x-tile.left] = Q if x % 2 == 0 else -Q
    return grid
