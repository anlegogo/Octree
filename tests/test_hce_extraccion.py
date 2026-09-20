from pathlib import Path

import numpy as np
import pytest

from hce_extraccion import (
    extraer_descriptores_hce,
    extraer_descriptores_hce_desde_npz,
    nombres_features,
    ocupacion_L0_verificacion,
)
from octree_real import construir_octree, guardar_octree_disperso
from validar_muestra_controlada_hce import (
    diagnostico_ocupacion_l1,
    estadisticas_features,
)


def nube_hce_controlada(seed=42, n=800):
    rng = np.random.default_rng(seed)
    puntos = rng.uniform(-0.95, 0.95, size=(n, 3)).astype(np.float32)
    normales = rng.normal(size=(n, 3)).astype(np.float32)
    normales /= np.clip(
        np.linalg.norm(normales, axis=1, keepdims=True), 1e-12, None,
    )
    return puntos, normales


@pytest.mark.parametrize(
    "profundidad,dimension", [(5, 17), (6, 18)],
)
def test_dimension_orden_e_integridad_hce(profundidad, dimension):
    puntos, normales = nube_hce_controlada()
    raiz = construir_octree(puntos, normales, profundidad)
    features = extraer_descriptores_hce(raiz, profundidad)
    nombres = nombres_features(profundidad)

    assert features.shape == (dimension,)
    assert len(nombres) == dimension
    assert len(set(nombres)) == dimension
    assert nombres[0] == "ocupacion_L1"
    assert "ocupacion_L0" not in nombres
    assert np.isfinite(features).all()
    assert ocupacion_L0_verificacion(raiz, profundidad) == 100.0

    coherencia_media = features[-2]
    coherencia_varianza = features[-1]
    assert 0.0 <= coherencia_media <= 1.0
    assert 0.0 <= coherencia_varianza <= 0.25


@pytest.mark.parametrize("profundidad", [5, 6])
def test_hce_es_reproducible_y_equivalente_al_npz(
    profundidad: int, tmp_path: Path,
):
    puntos, normales = nube_hce_controlada()
    raiz_a = construir_octree(puntos, normales, profundidad)
    raiz_b = construir_octree(puntos, normales, profundidad)
    features_a = extraer_descriptores_hce(raiz_a, profundidad)
    features_b = extraer_descriptores_hce(raiz_b, profundidad)
    np.testing.assert_array_equal(features_a, features_b)

    destino = tmp_path / f"octree_L{profundidad}.npz"
    guardar_octree_disperso(
        raiz_a, destino, etiqueta=0, profundidad_max=profundidad,
    )
    features_npz = extraer_descriptores_hce_desde_npz(str(destino))
    np.testing.assert_array_equal(features_a, features_npz)


def test_estadisticas_detectan_constantes_nan_e_inf():
    matriz = np.asarray([
        [100.0, 1.0, np.nan],
        [100.0, 2.0, np.inf],
        [100.0, 3.0, 5.0],
    ])
    estadisticas = estadisticas_features(matriz, ["ocupacion_L1", "x", "y"])

    assert estadisticas[0]["es_constante_finita"] is True
    assert estadisticas[0]["n_valores_unicos"] == 1
    assert estadisticas[1]["es_constante_finita"] is False
    assert estadisticas[2]["n_nan"] == 1
    assert estadisticas[2]["n_inf"] == 1


def test_decision_l1_exige_train_completo():
    estadisticas = [{
        "nombre": "ocupacion_L1",
        "min": 100.0,
        "max": 100.0,
        "media": 100.0,
        "mediana": 100.0,
        "desviacion_estandar": 0.0,
        "n_valores_unicos": 1,
        "n_nan": 0,
        "n_inf": 0,
        "es_constante_finita": True,
    }]

    parcial = diagnostico_ocupacion_l1(estadisticas, 5, False)
    completo = diagnostico_ocupacion_l1(estadisticas, 9843, True)
    assert parcial["estado"] == "evidencia_insuficiente"
    assert completo["estado"] == "constante_en_train_completo"
