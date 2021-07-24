from definitions import *
from kernels import *

def main():
    box = BoundingBox(0,10,0,10)
    grid = initialize_grid(box)
    print(grid)

    print("Left", box.left)

if __name__ == '__main__':
    main()
