import numpy as np
import pytest

from octree_real import construir_octree
from validacion_objetivo1 import validar_equivalencia_denso_octree


def nube_controlada(seed=2026, n=5000):
    rng = np.random.default_rng(seed)
    puntos = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    # Incluye explicitamente las fronteras y los planos de subdivision.
    puntos[:7] = np.asarray([
        [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, -1.0],
        [0.0, -1.0, 1.0],
    ], dtype=np.float32)
    normales = rng.normal(size=(n, 3)).astype(np.float32)
    normas = np.linalg.norm(normales, axis=1, keepdims=True)
    normales /= np.clip(normas, 1e-12, None)
    return puntos, normales


@pytest.mark.parametrize("resolucion,profundidad", [(32, 5), (64, 6)])
def test_equivalencia_celda_a_celda(resolucion, profundidad):
    puntos, normales = nube_controlada()
    raiz = construir_octree(puntos, normales, profundidad)
    resultado = validar_equivalencia_denso_octree(
        puntos, normales, raiz, resolucion,
    )
    assert resultado["equivalente"] is True
    assert resultado["n_celdas_denso"] == resultado["n_hojas_octree"]
