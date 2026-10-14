# cigaretSimulator

Simulador de fluidos incompresibles en 2D sobre malla MAC, escrito en Python con NumPy y Numba.

El método sigue el enfoque de Ten Minute Physics: proyección con Gauss-Seidel (red-black + SOR) y advección semi-lagrangiana de velocidades y humo.

Sale un cigarro que puedes mover con las flechas. Es un código antiguo que tenía de verano 2025, adaptado de un proyecto de primavera 2024 para Computación I de Física UAM. Ahora lo he corregido, optimizado, y he añadido el cigarro.


## Uso

```
python fluid_main.py
```

Para mover el cigarro hay que hacer clic una vez en la ventana para darle el foco. La primera ejecución tarda algo más porque Numba compila los kernels.

Los parámetros (tamaño de malla, dt, flotación, disipación del humo...) se ajustan en `fluid_main.py`. Con `cigarette=False` se usa un chorro fijo en lugar del cigarro.

La simulación corre en un proceso separado y la ventana solo pinta el último estado disponible, de modo que la animación no depende de lo que tarde cada paso.

## Estructura

- `fluid_grid.py`: malla (velocidades, humo, celdas sólidas) y lógica del cigarro
- `fluid_solver.py`: proyección, advección y paso de simulación
- `fluid_simulation.py`: ejecución de la simulación en un proceso aparte
- `fluid_visualization.py`: animación y gráficas
- `fluid_main.py`: punto de entrada y dos tests visuales (`test_divergence`, `test_quiver`)

## Pendiente

- `advect_v` usa la `u` ya advectada en lugar de la del inicio del paso
- Probar multigrid o gradiente conjugado, ya que la proyección es la mayor parte del coste
