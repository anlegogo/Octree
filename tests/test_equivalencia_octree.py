import numpy as np
import pytest

from cuantizacion import cuantizar_indices_octree
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


def nube_vecina_a_fronteras(resolucion):
    fronteras = np.linspace(
        -1.0, 1.0, resolucion + 1, dtype=np.float32,
    )[1:-1]
    debajo = np.nextafter(fronteras, np.float32(-np.inf))
    encima = np.nextafter(fronteras, np.float32(np.inf))

    bloques = []
    for eje in range(3):
        puntos = np.full((3 * len(fronteras), 3), 0.12345, dtype=np.float32)
        puntos[:, eje] = np.concatenate([debajo, fronteras, encima])
        bloques.append(puntos)

    puntos = np.concatenate(bloques)
    normales = np.tile(
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        (len(puntos), 1),
    )
    return puntos, normales


@pytest.mark.parametrize("resolucion", [32, 64])
def test_cuantizacion_conserva_lado_de_frontera(resolucion):
    fronteras = np.linspace(
        -1.0, 1.0, resolucion + 1, dtype=np.float32,
    )[1:-1]
    puntos = np.full((3 * len(fronteras), 3), 0.12345, dtype=np.float32)
    puntos[:, 0] = np.concatenate([
        np.nextafter(fronteras, np.float32(-np.inf)),
        fronteras,
        np.nextafter(fronteras, np.float32(np.inf)),
    ])

    indices = cuantizar_indices_octree(puntos, resolucion)[:, 0]
    celdas_superiores = np.arange(1, resolucion, dtype=np.int64)
    esperado = np.concatenate([
        celdas_superiores - 1,
        celdas_superiores,
        celdas_superiores,
    ])
    assert np.array_equal(indices, esperado)


@pytest.mark.parametrize("resolucion,profundidad", [(32, 5), (64, 6)])
def test_equivalencia_en_vecindad_de_fronteras(resolucion, profundidad):
    puntos, normales = nube_vecina_a_fronteras(resolucion)
    raiz = construir_octree(puntos, normales, profundidad)
    resultado = validar_equivalencia_denso_octree(
        puntos, normales, raiz, resolucion,
    )
    assert resultado["equivalente"] is True
