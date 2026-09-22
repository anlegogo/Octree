"""Pruebas geometricas del backend OctNet sin dependencia de PyTorch."""

import numpy as np
import pytest

from cuantizacion import cuantizar_indices_octree
from grid_octree import convertir_a_grid_octree
from octree_real import construir_octree


def _nube_controlada():
    puntos = np.asarray([
        [-0.91, -0.77, -0.63],
        [-0.12, 0.18, 0.37],
        [0.43, 0.61, 0.82],
        [0.88, -0.71, 0.09],
    ], dtype=np.float32)
    normales = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float32)
    normales /= np.linalg.norm(normales, axis=1, keepdims=True)
    return puntos, normales


def _expandir_hojas(geometria, atributos):
    canales = atributos.shape[1]
    volumen = np.empty(
        (canales, geometria.resolucion, geometria.resolucion,
         geometria.resolucion),
        dtype=np.float32,
    )
    for indice, (origen, tamano) in enumerate(
        zip(geometria.origenes, geometria.tamanos)
    ):
        x, y, z = (int(v) for v in origen)
        s = int(tamano)
        volumen[:, x:x + s, y:y + s, z:z + s] = atributos[indice, :, None, None, None]
    return volumen


@pytest.mark.parametrize("resolucion,profundidad", [(32, 5), (64, 6)])
def test_conversion_preserva_ocupacion_y_reconstruye_vacios(
    resolucion, profundidad,
):
    puntos, normales = _nube_controlada()
    raiz = construir_octree(puntos, normales, profundidad)
    muestra = convertir_a_grid_octree(raiz, resolucion)
    geometria = muestra.geometria

    assert geometria.forma_rejilla == (resolucion // 8,) * 3
    assert len(geometria.hojas_por_arbol) == (resolucion // 8) ** 3
    assert np.sum(geometria.tamanos.astype(np.int64) ** 3) == resolucion ** 3
    assert np.any((muestra.atributos[:, 0] == 0) & (geometria.tamanos > 1))

    ocupadas = muestra.atributos[:, 0] == 1
    assert np.all(geometria.tamanos[ocupadas] == 1)
    indices_esperados = cuantizar_indices_octree(puntos, resolucion)
    assert set(map(tuple, geometria.origenes[ocupadas])) == set(
        map(tuple, indices_esperados)
    )


@pytest.mark.parametrize("resolucion,profundidad", [(32, 5), (64, 6)])
def test_convolucion_dispersa_equivale_a_referencia_densa(
    resolucion, profundidad,
):
    puntos, normales = _nube_controlada()
    muestra = convertir_a_grid_octree(
        construir_octree(puntos, normales, profundidad_max=profundidad),
        resolucion,
    )
    geometria = muestra.geometria
    rng = np.random.default_rng(42)
    entrada = rng.normal(size=(geometria.n_hojas, 2)).astype(np.float32)
    pesos = rng.normal(size=(3, 2, 3, 3, 3)).astype(np.float32)
    bias = rng.normal(size=3).astype(np.float32)

    plan = geometria.plan_convolucion
    salida_dispersa = np.broadcast_to(
        bias, (geometria.n_hojas, 3),
    ).copy()
    pesos_planos = pesos.reshape(3, 2, 27)
    for kernel in range(27):
        mascara = plan.kernel == kernel
        contribucion = (
            entrada[plan.entrada[mascara]] @ pesos_planos[:, :, kernel].T
        ) * plan.coeficiente[mascara, None]
        np.add.at(salida_dispersa, plan.salida[mascara], contribucion)

    volumen = _expandir_hojas(geometria, entrada)
    acolchado = np.pad(volumen, ((0, 0), (1, 1), (1, 1), (1, 1)))
    salida_densa = np.broadcast_to(
        bias[:, None, None, None],
        (3, resolucion, resolucion, resolucion),
    ).copy()
    for kx in range(3):
        for ky in range(3):
            for kz in range(3):
                ventana = acolchado[
                    :,
                    kx:kx + resolucion,
                    ky:ky + resolucion,
                    kz:kz + resolucion,
                ]
                salida_densa += np.einsum(
                    "oi,ixyz->oxyz", pesos[:, :, kx, ky, kz], ventana,
                )

    esperada = np.empty_like(salida_dispersa)
    for indice, (origen, tamano) in enumerate(
        zip(geometria.origenes, geometria.tamanos)
    ):
        x, y, z = (int(v) for v in origen)
        s = int(tamano)
        esperada[indice] = salida_densa[
            :, x:x + s, y:y + s, z:z + s,
        ].mean(axis=(1, 2, 3))
    np.testing.assert_allclose(salida_dispersa, esperada, rtol=2e-5, atol=2e-5)


def test_pooling_disperso_equivale_a_maxpool_denso():
    puntos, normales = _nube_controlada()
    geometria = convertir_a_grid_octree(
        construir_octree(puntos, normales, profundidad_max=5), 32,
    ).geometria
    rng = np.random.default_rng(7)
    entrada = rng.normal(size=(geometria.n_hojas, 3)).astype(np.float32)
    plan = geometria.plan_pooling

    salida = np.full((plan.geometria_salida.n_hojas, 3), -np.inf, np.float32)
    np.maximum.at(salida, plan.entrada_a_salida, entrada)

    volumen = _expandir_hojas(geometria, entrada)
    denso_pool = volumen.reshape(3, 16, 2, 16, 2, 16, 2).max(
        axis=(2, 4, 6),
    )
    for indice, (origen, tamano) in enumerate(zip(
        plan.geometria_salida.origenes,
        plan.geometria_salida.tamanos,
    )):
        x, y, z = (int(v) for v in origen)
        s = int(tamano)
        bloque = denso_pool[:, x:x + s, y:y + s, z:z + s]
        esperado = bloque[:, 0, 0, 0]
        np.testing.assert_allclose(
            bloque,
            np.broadcast_to(esperado[:, None, None, None], bloque.shape),
        )
        np.testing.assert_allclose(salida[indice], esperado)
