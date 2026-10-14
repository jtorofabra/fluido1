import multiprocessing as mp
import os
import time
from multiprocessing import shared_memory

import numpy as np

PUBLISH_HZ = 120


def _apply_source(grid, solver, cigarette, pos):
    # pos = [empuje_x, empuje_y, j0, i0], en memoria compartida con el proceso principal
    if not cigarette:
        grid.add_source()
        return
    grid.update_cigarette(pos[0], pos[1], solver.dt)
    pos[3], pos[2] = grid.cig_cell


def _run_simulation(grid, solver, shm_name, lock, step_count, sps, stop,
                    steps_per_second, max_steps, cigarette, pos):
    parent_pid = os.getppid()
    shm = shared_memory.SharedMemory(name=shm_name)
    smoke_out = np.ndarray(grid.smoke.shape, dtype=grid.smoke.dtype, buffer=shm.buf)

    period = 1.0 / steps_per_second if steps_per_second else 0.0
    next_t = time.perf_counter()
    last_t = next_t
    last_publish = 0.0
    measured = 0.0
    n = 0

    def publish():
        # Si la visualización tiene el lock, se publica en el siguiente paso
        if lock.acquire(block=False):
            try:
                np.copyto(smoke_out, grid.smoke)
            finally:
                lock.release()
            return True
        return False

    try:
        while not stop.is_set():
            if max_steps is not None and n >= max_steps:
                break
            # Salir si el proceso principal ha muerto, para no dejar un huérfano
            if os.getppid() != parent_pid:
                return

            _apply_source(grid, solver, cigarette, pos)
            solver.general_step()
            n += 1
            step_count.value = n

            # Limitado a PUBLISH_HZ: publicar en cada paso acaparaba el lock y bloqueaba la visualización
            now = time.perf_counter()
            if now - last_publish >= 1.0 / PUBLISH_HZ and publish():
                last_publish = now

            measured = 0.9 * measured + 0.1 / max(now - last_t, 1e-9)
            sps.value = measured
            last_t = now

            if period:
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    stop.wait(delay)
                else:
                    next_t = time.perf_counter()
        with lock:
            np.copyto(smoke_out, grid.smoke)
    finally:
        del smoke_out
        shm.close()


class FluidSimulation:
    # La simulación corre en un proceso aparte y no en un hilo porque Numba y
    # matplotlib compiten por el GIL (en 512x512 la proyección pasaba de 26 ms a 1.5 s).
    # Ojo: el proceso hijo trabaja con su propia copia de grid y solver.

    def __init__(self, grid, solver, steps_per_second=None, max_steps=None,
                 cigarette=False):
        self.grid = grid
        self.solver = solver
        self.steps_per_second = steps_per_second
        self.max_steps = max_steps
        self.cigarette = cigarette

        # 'spawn' porque fork con los hilos de Numba no es seguro
        self._ctx = mp.get_context('spawn')
        self._lock = self._ctx.Lock()
        self._stop = self._ctx.Event()
        self._step_count = self._ctx.Value('q', 0, lock=False)
        self._sps = self._ctx.Value('d', 0.0, lock=False)
        self._pos = self._ctx.Array('d', [0.0, 0.0, -1, -1], lock=False)

        self._shm = shared_memory.SharedMemory(create=True, size=grid.smoke.nbytes)
        self._smoke = np.ndarray(grid.smoke.shape, dtype=grid.smoke.dtype, buffer=self._shm.buf)
        self._smoke[:] = grid.smoke
        self._process = None

    @property
    def step_count(self):
        return self._step_count.value

    @property
    def measured_sps(self):
        return self._sps.value

    def set_thrust(self, ax, ay):
        self._pos[0], self._pos[1] = ax, ay

    def cigarette_cell(self):
        if self._pos[2] < 0:
            return None
        return int(self._pos[3]), int(self._pos[2])

    def step(self):
        # Un paso en el proceso actual, para pruebas sin start()
        _apply_source(self.grid, self.solver, self.cigarette, self._pos)
        self.solver.general_step()
        np.copyto(self._smoke, self.grid.smoke)
        self._step_count.value += 1

    def latest_smoke(self):
        with self._lock:
            return self._smoke.copy()

    def start(self):
        self._process = self._ctx.Process(
            target=_run_simulation, daemon=True,
            args=(self.grid, self.solver, self._shm.name, self._lock,
                  self._step_count, self._sps, self._stop,
                  self.steps_per_second, self.max_steps, self.cigarette, self._pos))
        self._process.start()

    def join(self):
        if self._process is not None:
            self._process.join()

    def stop(self):
        self._stop.set()
        self.join()
        if self._shm is not None:
            del self._smoke
            self._shm.close()
            self._shm.unlink()
            self._shm = None
