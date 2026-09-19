from pathlib import Path

import numpy as np
import pytest

from octree_real import (
    OCTREE_FORMAT_VERSION,
    _serializar_dfs,
    carga_binaria_sin_comprimir,
    construir_octree,
    guardar_octree_disperso,
    leer_metadatos_octree,
    reconstruir_octree_desde_npz,
)
from validacion_objetivo1 import validar_equivalencia_denso_octree


def nube_simple():
    puntos = np.asarray([
        [-0.9, -0.8, -0.7], [-0.2, 0.1, 0.3], [0.4, 0.6, 0.8],
        [0.99, -0.99, 0.0], [0.0, 0.0, 0.0],
    ], dtype=np.float32)
    normales = np.asarray([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    return puntos, normales


def test_roundtrip_con_metadatos_y_jerarquia(tmp_path: Path):
    puntos, normales = nube_simple()
    raiz = construir_octree(puntos, normales, profundidad_max=5)
    destino = tmp_path / "objeto.npz"
    guardar_octree_disperso(
        raiz, destino, etiqueta=8, profundidad_max=5,
        metadatos={
            "model_id": "chair_control",
            "categoria": "chair",
            "split": "train",
            "semilla_muestreo": 42,
            "n_puntos_muestreo": len(puntos),
            "archivo_origen_sha256": "a" * 64,
        },
    )

    metadatos = leer_metadatos_octree(destino)
    assert metadatos["format_version"] == OCTREE_FORMAT_VERSION
    assert metadatos["model_id"] == "chair_control"
    assert metadatos["resolucion"] == 32

    raiz_cargada, etiqueta, profundidad = reconstruir_octree_desde_npz(destino)
    assert etiqueta == 8
    assert profundidad == 5
    assert validar_equivalencia_denso_octree(
        puntos, normales, raiz_cargada, 32,
    )["equivalente"]
    for original, cargado in zip(
        _serializar_dfs(raiz), _serializar_dfs(raiz_cargada),
    ):
        np.testing.assert_array_equal(original, cargado)


def test_payload_binario_incluye_coherencia():
    puntos, normales = nube_simple()
    raiz = construir_octree(puntos, normales, profundidad_max=5)
    carga = carga_binaria_sin_comprimir(raiz)
    assert carga["bytes_por_nodo"] == 18
    assert carga["total_bytes"] % 18 == 0


def test_rechaza_formato_antiguo(tmp_path: Path):
    destino = tmp_path / "antiguo.npz"
    np.savez_compressed(
        destino,
        profundidades=np.asarray([0], dtype=np.uint8),
        mascaras=np.asarray([0], dtype=np.uint8),
        normales=np.zeros((1, 3), dtype=np.float32),
        coherencias=np.zeros(1, dtype=np.float32),
        etiqueta=0,
        profundidad_max=0,
    )
    with pytest.raises(ValueError, match="antiguo o incompleto"):
        reconstruir_octree_desde_npz(destino)
