import numpy as np

class FluidGrid:
    def __init__(self, nx, ny, dx):
        self.nx = nx
        self.ny = ny
        self.dx = dx

        self.u = np.zeros((ny, nx+1), dtype=np.float32)  # x-component of velocity
        self.v = np.zeros((ny+1, nx), dtype=np.float32)  # y-component of velocity
        self.temp_u = np.zeros((ny, nx+1), dtype=np.float32)
        self.temp_v = np.zeros((ny+1, nx), dtype=np.float32)

        self.smoke=np.zeros((ny, nx), dtype=np.float32)
        self.temp_smoke = np.zeros((ny, nx), dtype=np.float32)
        self.divergence = np.zeros((ny, nx), dtype=np.float32)

        #Marca que casillas son accesibles (1) o no (0)
        self.cell_type = np.ones((ny, nx), dtype=np.float32)

        # Hacer bordes no accesibles (0)
        self.cell_type[0, :] = 0    # Borde inferior
        self.cell_type[-1, :] = 0   # Borde superior
        self.cell_type[:, 0] = 0    # Borde izquierdo
        self.cell_type[:, -1] = 0   # Borde derecho

        self._build_face_masks_and_weights()

        # Cigarro: obstáculo móvil con la fuente de humo en la punta
        self.cig_len = max(8, nx // 4)
        self.cig_th = max(2, self.cig_len // 8)
        self.cig_ember_offset = max(1, self.cig_th // 2)  # celdas entre la brasa y la punta
        self.cig_max_speed = max(0.5, 0.01 * nx)
        self.cig_friction = 0.05
        self.cig_rise = 0.5  # velocidad vertical del humo en la brasa
        self.cig_x = None
        self.cig_y = None
        self.cig_vx = 0.0
        self.cig_vy = 0.0
        self.cig_cell = None  # (i0, j0): celda inferior izquierda
        self._base_cell_type = self.cell_type.copy()

    def customtest(self):
        self.u = np.zeros((self.ny, self.nx+1))  # x-component of velocity
        self.v = np.zeros((self.ny+1, self.nx))  # y-component of velocity
        self.u[20:30, 20:30]=1

    def add_source(self):
        self.u[20:24,5:10]=3
        self.smoke[20:24,5:10]=1

    def update_cigarette(self, ax, ay, dt):
        # (ax, ay) en {-1, 0, 1}: dirección de empuje según las flechas.
        # El filtro y el papel son sólidos y empujan el fluido; la brasa emite humo.
        L, T = self.cig_len, self.cig_th
        E = max(1, T)
        jb = L - E - self.cig_ember_offset  # inicio de la brasa, relativo a j0
        x_min, x_max = 1 + L / 2, self.nx - 1 - L / 2
        y_min, y_max = 1 + T / 2, self.ny - 1 - T / 2

        if self.cig_x is None:
            self.cig_x, self.cig_y = self.nx / 2, self.ny / 2

        # Empuje y rozamiento, con la velocidad limitada a vmax
        f, vmax = self.cig_friction, self.cig_max_speed
        self.cig_vx += f * vmax * ax * dt
        self.cig_vy += f * vmax * ay * dt
        damp = max(0.0, 1.0 - f * dt)
        self.cig_vx *= damp
        self.cig_vy *= damp
        speed = np.hypot(self.cig_vx, self.cig_vy)
        if speed > vmax:
            self.cig_vx *= vmax / speed
            self.cig_vy *= vmax / speed

        x = self.cig_x + self.cig_vx * dt
        y = self.cig_y + self.cig_vy * dt
        if not x_min <= x <= x_max:
            x = min(max(x, x_min), x_max); self.cig_vx = 0.0
        if not y_min <= y <= y_max:
            y = min(max(y, y_min), y_max); self.cig_vy = 0.0
        self.cig_x, self.cig_y = x, y
        vx, vy = self.cig_vx, self.cig_vy

        i0 = int(round(y - T / 2))
        j0 = int(round(x - L / 2))
        js = j0 + jb
        if (i0, j0) != self.cig_cell:
            self.cig_cell = (i0, j0)
            self.cell_type[:] = self._base_cell_type
            self.cell_type[i0:i0+T, j0:js] = 0
            self._build_face_masks_and_weights()
            self.smoke[i0:i0+T, j0:js] = 0

        self.u[i0:i0+T, j0:js+1] = vx
        self.v[i0:i0+T+1, j0:js] = vy

        # Fuente de humo en la brasa
        self.smoke[i0:i0+T, js:js+E] = 1
        self.v[i0+1:i0+T+1, js:js+E] = vy + self.cig_rise

    def _build_face_masks_and_weights(self):
        ct = self.cell_type.astype(np.float32)
        ctp = np.pad(ct, 1, constant_values=0)

        self.v_mask = ctp[0:-1, 1:-1] * ctp[1:, 1:-1]   # (ny+1, nx)

        cC = ctp[1:-1, 1:-1]; cL = ctp[1:-1, 0:-2]; cR = ctp[1:-1, 2:]
        cB = ctp[0:-2, 1:-1]; cT = ctp[2:,   1:-1]
        self.wL = cC * cL; self.wR = cC * cR; self.wB = cC * cB; self.wT = cC * cT
        self.s  = self.wL + self.wR + self.wB + self.wT

        # Coeficientes w/s precalculados para no dividir en el solver
        inv_s = np.divide(1.0, self.s, out=np.zeros_like(self.s), where=self.s > 0)
        self.cL = self.wL * inv_s; self.cR = self.wR * inv_s
        self.cB = self.wB * inv_s; self.cT = self.wT * inv_s
