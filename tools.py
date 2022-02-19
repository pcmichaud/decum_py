import numpy as np
from numba import njit

@njit(fastmath=True)
def scale(x,grid):
    nx = len(grid)
    ilow = nx-1
    iup = nx-1
    xu = 0.0
    for i in range(nx):
        if x <= grid[i]:
            ilow = i-1
            if ilow<0:
                ilow = 0
            iup = i
            if ilow!=iup:
                xu = (x-grid[ilow])/(grid[iup]- grid[ilow])
            else :
                xu = 0.0
            break
    return ilow,iup,xu

@njit(fastmath=True)
def interp2d(r,s,xm):
    px = xm[0,0] + r*(-xm[0,0]+xm[1,0]) + s*(-xm[0,0]+xm[0,1]) \
         + r*s*(xm[0,0] - xm[1,0] - xm[0,1] + xm[1,1])
    return px

