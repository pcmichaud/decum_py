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

@njit(fastmath=True)
def cubic_kernel(t):
    """Cubic interpolation kernel."""
    t = abs(t)
    if t <= 1:
        return 1.5 * t**3 - 2.5 * t**2 + 1
    elif t <= 2:
        return -0.5 * t**3 + 2.5 * t**2 - 4 * t + 2
    else:
        return 0.0

@njit(fastmath=True)
def cubic_interp1d(x, y, xi):
    """
    Perform cubic interpolation in 1D.
    :param x: Array of grid points (assumed equidistant).
    :param y: Array of values at grid points.
    :param xi: Interpolation point.
    :return: Interpolated value.
    """
    dx = x[1] - x[0]  # Grid spacing
    i = int((xi - x[0]) / dx)  # Index of the closest grid point to the left
    t = (xi - x[i]) / dx  # Fractional distance from x[i]

    # Ensure the index range is within bounds
    i0 = max(i - 1, 0)
    i1 = i
    i2 = min(i + 1, len(x) - 1)
    i3 = min(i + 2, len(x) - 1)

    # Compute interpolated value using cubic kernel
    value = y[i0] * cubic_kernel(t + 1) + y[i1] * cubic_kernel(t) + y[i2] * cubic_kernel(t - 1) + y[i3] * cubic_kernel(t - 2)
    return value

@njit
def cubic_interp2d(x, y, z, xi, yi):
    """
    Perform cubic interpolation in 2D.
    :param x: 1D array of x-coordinates (grid points).
    :param y: 1D array of y-coordinates (grid points).
    :param z: 2D array of values on the grid.
    :param xi: x-coordinate for interpolation.
    :param yi: y-coordinate for interpolation.
    :return: interpolated value.
    """
    # Interpolate along x for the current row
    temp_values = np.zeros(len(x))
    for k in range(len(x)):
        temp_values[k] = cubic_interp1d(y, z[:, k], yi)
    # Interpolate along y for the current column
    zi = cubic_interp1d(x, temp_values, xi)
    return zi


