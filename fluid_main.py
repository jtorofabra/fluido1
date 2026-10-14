from fluid_grid import FluidGrid
from fluid_solver import FluidSolver
from fluid_visualization import FluidVisualization
from fluid_simulation import FluidSimulation

import matplotlib.pyplot as plt

def test_divergence(grid, viz, solver):
    grid.customtest()  # Reinicializar para el test
    # Crear figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Figura 1: Divergencia inicial
    viz.plot_divergence(solver, title="Divergencia Inicial", ax=ax1)
    solver.solve(iterations=30)  # Ejecutar el solver
    # Figura 2: Divergencia final
    viz.plot_divergence(solver, title="Divergencia Final", ax=ax2)

    plt.tight_layout()
    plt.show()

def test_quiver(grid, viz, solver):
    grid.customtest()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    viz.plot_velocity_field(title="Velocidades Iniciales", ax=ax1)

    solver.solve(iterations=20)

    viz.plot_velocity_field(title="Velocidades (sin divergencia)", ax=ax2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    grid = FluidGrid(nx=512, ny=512, dx=1.0)
    viz = FluidVisualization(grid)
    solver = FluidSolver(grid, over_relaxation=1.75, dt=1, buoyancy=0.09, smoke_decay=0.001)

    # steps_per_second=None: sin límite de velocidad
    # cigarette=False: chorro fijo en lugar del cigarro
    sim = FluidSimulation(grid, solver, steps_per_second=60, max_steps=None, cigarette=True)
    sim.start()
    try:
        viz.animate_smoke(sim, fps=60, title="Cigarette Simulator")
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop()
