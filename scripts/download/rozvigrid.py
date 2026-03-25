from rozvidrought_datasets.grid import RozviGrid

grid = RozviGrid()

# corner pixels
top_left = grid.pixel_bounds(0)
bottom_right = grid.pixel_bounds(grid.width * grid.height - 1)

west = top_left[0]
north = top_left[3]
east = bottom_right[2]
south = bottom_right[1]

area = [north, west, south, east]

print("Rozvi bounds:", (west, south, east, north))
print("CDS area:", area)