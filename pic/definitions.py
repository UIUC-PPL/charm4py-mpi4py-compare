from numba.experimental import jitclass
import numba
import numpy as np

PRK_M_PI = 3.14159265358979323846264338327950288419716939937510
MASS_INV = 1.0
Q = 1.0
epsilon = 0.000001
DT = 1.0
MEMORYSLACK = 10

REL_X = 0.5
REL_Y = 0.5

GEOMETRIC = 10
SINUSOIDAL = 11
LINEAR = 12
PATCH = 13
UNDEFINED = 14

PARTICLE_FIELDS = 10
PARTICLE_X = 0
PARTICLE_Y = 1
PARTICLE_VX = 2
PARTICLE_VY = 3
PARTICLE_Q = 4
PARTICLE_X0 = 5
PARTICLE_Y0 = 6
PARTICLE_K = 7
PARTICLE_M = 8
PARTICLE_ID = 9

@jitclass
class BoundingBox:
    left: int
    right: int
    bottom: int
    top: int

    def __init__(self, left: int, right: int, bottom: int, top: int):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
