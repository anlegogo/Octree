"""Cuantizacion espacial consistente con la subdivision recursiva del octree."""

from __future__ import annotations

import numpy as np


def cuantizar_indices_octree(
    coordenadas: np.ndarray, resolucion: int,
) -> np.ndarray:
    """Asigna coordenadas a celdas usando la convencion del octree.

    Cada frontera interna pertenece a la celda superior, igual que la
    comparacion ``>=`` usada al descender por el arbol. Se usa una busqueda
    explicita de fronteras porque la expresion aritmetica equivalente en
    ``float32`` puede redondear hacia arriba puntos situados apenas por debajo
    de una frontera y producir una asignacion distinta a la del octree.
    """
    valores = np.asarray(coordenadas)
    if valores.ndim != 2 or valores.shape[1] != 3:
        raise ValueError("Las coordenadas deben tener forma (N, 3)")

    resolucion = int(resolucion)
    if resolucion <= 0:
        raise ValueError("La resolucion debe ser un entero positivo")

    fronteras = np.linspace(
        -1.0, 1.0, resolucion + 1, dtype=np.float64,
    )[1:-1]
    indices = np.empty(valores.shape, dtype=np.int64)
    for eje in range(3):
        indices[:, eje] = np.searchsorted(
            fronteras, valores[:, eje], side="right",
        )

    return np.clip(indices, 0, resolucion - 1)
