import numpy as np
from numba import njit, prange

# Kernels compilados con Numba. Cada uno se compila en serie y en paralelo;
# la versión paralela solo compensa en mallas grandes.
PARALLEL_MIN_CELLS = 300 * 300  # umbral medido empíricamente


def _kernel(fn):
    # La versión paralela va sin cache: si no, ambas versiones se pisan en la cache de Numba
    opts = dict(nogil=True, fastmath=True)
    return njit(cache=True, **opts)(fn), njit(parallel=True, **opts)(fn)


@njit(cache=True, nogil=True, fastmath=True)
def _sample(field, x, y, dx):
    # Interpolación bilineal de field en el punto (x, y)
    ny, nx = field.shape
    x0 = int(min(max(np.floor(x / dx), 0), nx - 2))
    y0 = int(min(max(np.floor(y / dx), 0), ny - 2))
    sx = (x - x0 * dx) / dx
    sy = (y - y0 * dx) / dx
    return (field[y0,     x0] * (1.0 - sx) * (1.0 - sy) +
            field[y0,     x0 + 1] * sx * (1.0 - sy) +
            field[y0 + 1, x0] * (1.0 - sx) * sy +
            field[y0 + 1, x0 + 1] * sx * sy)


def _project(u, v, cL, cR, cB, cT, s, over_relaxation, iterations):
    # Gauss-Seidel red-black con SOR: primero las celdas con (i+j) par y luego
    # las impares, para poder paralelizar cada color.
    ny, nx = s.shape
    for _ in range(iterations):
        for color in range(2):
            for i in prange(ny):
                for j in range((i + color) % 2, nx, 2):
                    if s[i, j] <= 0:
                        continue

                    d = (
                        - u[i, j] + u[i, j+1]
                        - v[i, j] + v[i+1, j]
                    ) * over_relaxation

                    u[i, j]   += d * cL[i, j]
                    u[i, j+1] -= d * cR[i, j]
                    v[i, j]   += d * cB[i, j]
                    v[i+1, j] -= d * cT[i, j]


def _advect_u(u, v, temp_u, dx, dt):
    ny = u.shape[0]
    nx = u.shape[1] - 1
    temp_u[:] = u
    for i in prange(ny):
        for j in range(1, nx):
            x0 = dx * j
            y0 = dx * (i + 0.5)
            u0 = u[i, j]
            v0 = (v[i, j-1] + v[i+1, j-1] + v[i, j] + v[i+1, j]) * 0.25
            x1 = x0 - dt * u0
            y1 = y0 - dt * v0
            temp_u[i, j] = _sample(u, x1, y1 - 0.5 * dx, dx)
    u[:] = temp_u


def _advect_v(u, v, temp_v, dx, dt):
    ny = v.shape[0] - 1
    nx = v.shape[1]
    temp_v[:] = v
    for i in prange(1, ny):
        for j in range(0, nx):
            x0 = dx * (j + 0.5)
            y0 = dx * i
            u0 = (u[i-1, j] + u[i-1, j+1] + u[i, j] + u[i, j+1]) * 0.25
            v0 = v[i, j]
            x1 = x0 - dt * u0
            y1 = y0 - dt * v0
            temp_v[i, j] = _sample(v, x1 - 0.5 * dx, y1, dx)
    v[:] = temp_v


def _advect_smoke(u, v, smoke, temp_smoke, cell_type, dx, dt):
    ny, nx = smoke.shape
    temp_smoke[:] = smoke
    for i in prange(ny):
        for j in range(nx):
            if cell_type[i, j] == 0:
                continue
            x0 = dx * (j + 0.5)
            y0 = dx * (i + 0.5)
            u_center = 0.5 * (u[i, j] + u[i, j+1])
            v_center = 0.5 * (v[i, j] + v[i+1, j])
            x1 = x0 - dt * u_center
            y1 = y0 - dt * v_center
            temp_smoke[i, j] = _sample(smoke, x1 - 0.5 * dx, y1 - 0.5 * dx, dx)
    smoke[:] = temp_smoke


_PROJECT = _kernel(_project)
_ADVECT_U = _kernel(_advect_u)
_ADVECT_V = _kernel(_advect_v)
_ADVECT_SMOKE = _kernel(_advect_smoke)


class FluidSolver:
    def __init__(self, grid, over_relaxation=1.0, dt=0.1, parallel=None,
                 buoyancy=0.0, smoke_decay=0.0):
        self.grid = grid
        self.over_relaxation = over_relaxation
        self.dt = dt
        self.buoyancy = buoyancy
        self.smoke_decay = smoke_decay
        if parallel is None:
            parallel = grid.nx * grid.ny >= PARALLEL_MIN_CELLS
        self.parallel = parallel

    def _k(self, kernel):
        return kernel[1] if self.parallel else kernel[0]

    def calculate_divergence(self):
        dx = self.grid.dx
        u = self.grid.u    # shape (ny, nx+1)
        v = self.grid.v    # shape (ny+1, nx)
        du_dx = (u[:, 1:] - u[:, :-1]) / dx
        dv_dy = (v[1:, :] - v[:-1, :]) / dx
        np.copyto(self.grid.divergence, du_dx + dv_dy)

    def solve(self, iterations=70):
        g = self.grid
        self._k(_PROJECT)(g.u, g.v, g.cL, g.cR, g.cB, g.cT, g.s,
                          self.over_relaxation, iterations)

    def advect_u(self):
        g = self.grid
        self._k(_ADVECT_U)(g.u, g.v, g.temp_u, g.dx, self.dt)

    def advect_v(self):
        g = self.grid
        self._k(_ADVECT_V)(g.u, g.v, g.temp_v, g.dx, self.dt)

    def advect_smoke(self):
        g = self.grid
        self._k(_ADVECT_SMOKE)(g.u, g.v, g.smoke, g.temp_smoke, g.cell_type, g.dx, self.dt)

    def add_gravity(self, g=9.81):
        self.grid.v[:, :] -= g * self.dt

    def add_buoyancy(self):
        # El humo sube; solo se aplica en caras con fluido a ambos lados
        s = self.grid.smoke
        self.grid.v[1:-1, :] += ((self.buoyancy * self.dt * 0.5) * (s[:-1, :] + s[1:, :])
                                 * self.grid.v_mask[1:-1, :])

    def decay_smoke(self):
        self.grid.smoke *= max(0.0, 1.0 - self.smoke_decay * self.dt)

    def enforce_boundary_conditions(self):
        self.grid.u[:, 0] = 0.0
        self.grid.u[:, -1] = 0.0
        self.grid.v[0, :] = 0.0
        self.grid.v[-1, :] = 0.0

    def general_step(self):
        #self.add_gravity()
        if self.buoyancy:
            self.add_buoyancy()
        self.solve(iterations=100)
        self.enforce_boundary_conditions()
        self.advect_u()
        self.advect_v()
        self.enforce_boundary_conditions()
        self.advect_smoke()
        if self.smoke_decay:
            self.decay_smoke()
