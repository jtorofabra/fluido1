import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

class FluidVisualization:

    def __init__(self, grid):
        self.grid = grid

    def get_velocity_field(self):
        u_center = 0.5 * (self.grid.u[:, 0:-1] + self.grid.u[:, 1:])   # (ny, nx)
        v_center = 0.5 * (self.grid.v[0:-1, :] + self.grid.v[1:, :])   # (ny, nx)

        return u_center, v_center

    def plot_velocity_field(self, title="Velocity Field", ax=None):
        u_combined, v_combined = self.get_velocity_field()

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            show_now = True
        else:
            show_now = False

        ax.clear()

        ax.quiver(u_combined, v_combined, scale=10)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if show_now:
            plt.show()

    def plot_divergence(self, solver, title="Divergencia del Campo de Velocidades", ax=None):
        solver.calculate_divergence()
        divergence = self.grid.divergence.copy()

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            show_now = True
        else:
            show_now = False

        im = ax.imshow(divergence, cmap='RdBu', origin='lower')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)

        div_max = np.max(np.abs(divergence))
        ax.text(0.02, 0.98, f'Div máx: {div_max:.6f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        if show_now:
            plt.show()

        return divergence

    def animate_smoke(self, sim, fps=60, title="Smoke Field", max_pixels=768,
                      cigarette_image="cigarette.png"):
        ny, nx = self.grid.ny, self.grid.nx
        # En mallas grandes se pinta 1 de cada k celdas (la ventana no tiene más resolución)
        k = max(1, -(-max(ny, nx) // max_pixels))

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(sim.latest_smoke()[::k, ::k], cmap='gray', origin='lower',
                       vmin=0.0, vmax=1.0, interpolation='nearest', animated=True,
                       extent=(-0.5, nx - 0.5, -0.5, ny - 0.5))
        ax.set_title(title if k == 1 else f"{title} (1 de cada {k} celdas)")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        status = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top',
                         color='white', animated=True)
        artists = [im, status]

        if sim.cigarette:
            L, T = self.grid.cig_len, self.grid.cig_th
            if cigarette_image and os.path.exists(cigarette_image):
                img = plt.imread(cigarette_image)
                if img.ndim == 3 and img.shape[2] == 4:
                    # Recortar los márgenes transparentes para que coincida con el obstáculo
                    opaque = img[..., 3] > 0.95
                    rows = np.where(opaque.sum(axis=1) > 0.02 * img.shape[1])[0]
                    cols = np.where(opaque[rows].sum(axis=0) > 0.02 * max(rows.size, 1))[0] \
                        if rows.size else rows
                    if rows.size and cols.size:
                        img = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
                cig = [ax.imshow(img, origin='upper',
                                 extent=(0, L, 0, T), animated=True, zorder=3)]

                def place(i0, j0):
                    cig[0].set_extent((j0 - 0.5, j0 + L - 0.5, i0 - 0.5, i0 + T - 0.5))
            else:
                # Sin imagen: filtro, papel y brasa con rectángulos
                f_len, e_len = 0.25 * L, max(1.0, 0.08 * L)
                cig = [Rectangle((0, 0), f_len, T, color='#c8813a', animated=True, zorder=3),
                       Rectangle((0, 0), L - f_len - e_len, T, color='#f2f0ea', animated=True, zorder=3),
                       Rectangle((0, 0), e_len, T, color='#ff5a1f', animated=True, zorder=3)]
                offsets = [0.0, f_len, L - e_len]
                for r in cig:
                    ax.add_patch(r)

                def place(i0, j0):
                    for r, off in zip(cig, offsets):
                        r.set_xy((j0 - 0.5 + off, i0 - 0.5))
            for a in cig:
                a.set_visible(False)
            artists += cig

            # Quitar las flechas de los atajos de teclado de matplotlib
            for name in [k for k in plt.rcParams if k.startswith('keymap.')]:
                plt.rcParams[name] = [key for key in plt.rcParams[name]
                                      if key not in ('left', 'right', 'up', 'down')]
            held = set()
            arrows = {'left', 'right', 'up', 'down'}

            def send_thrust():
                sim.set_thrust(('right' in held) - ('left' in held),
                               ('up' in held) - ('down' in held))

            def on_key(event):
                key = (event.key or '').split('+')[-1]  # 'shift+left' -> 'left'
                if key in arrows:
                    (held.add if event.name == 'key_press_event' else held.discard)(key)
                    send_thrust()
            fig.canvas.mpl_connect('key_press_event', on_key)
            fig.canvas.mpl_connect('key_release_event', on_key)
            ax.set_title(ax.get_title() + "\n(flechas para mover el cigarro)", fontsize=10)

        # El imshow del cigarro puede haber cambiado los límites
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(-0.5, ny - 0.5)

        def update(_):
            im.set_data(sim.latest_smoke()[::k, ::k])
            status.set_text(f'paso {sim.step_count} - {sim.measured_sps:.0f} pasos/s')
            if sim.cigarette:
                cell = sim.cigarette_cell()
                if cell is not None:
                    place(*cell)
                    for a in cig:
                        a.set_visible(True)
            return artists

        anim = FuncAnimation(fig, update, interval=1000 / fps, blit=True,
                             cache_frame_data=False)
        plt.show()
        return anim
