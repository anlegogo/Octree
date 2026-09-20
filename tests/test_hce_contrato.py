from pathlib import Path

import numpy as np
import pytest

from auditar_train_hce import auditar_resolucion
from contrato_hce import (
    aplicar_contrato_hce,
    construir_contrato_hce,
    estadisticas_features,
    validar_contrato_hce,
)
from hce_extraccion import nombres_features
from octree_real import construir_octree, guardar_octree_disperso
from preprocesar_octrees import CLASE_A_INDICE


def test_contrato_completo_excluye_constantes_y_se_aplica():
    nombres = nombres_features(5)
    matriz = np.tile(np.arange(len(nombres), dtype=np.float32), (4, 1))
    matriz[:, 1] = [1.0, 2.0, 3.0, 4.0]
    matriz[:, 2] = [0.0, 1.0, 0.0, 1.0]
    estadisticas = estadisticas_features(matriz, nombres)

    contrato = construir_contrato_hce(
        resolucion=32,
        profundidad_max=5,
        estadisticas=estadisticas,
        n_modelos_train=4,
        conteo_train_esperado=4,
        auditoria_train_completa=True,
    )

    validar_contrato_hce(contrato, resolucion=32)
    assert contrato["autorizado_para_entrenamiento"] is True
    assert contrato["indices_incluidos"] == [1, 2]
    assert contrato["orden_modelo"] == nombres[1:3]
    seleccion = aplicar_contrato_hce(matriz, contrato)
    np.testing.assert_array_equal(seleccion, matriz[:, 1:3])


def test_contrato_incompleto_bloquea_entrenamiento():
    nombres = nombres_features(5)
    matriz = np.arange(2 * len(nombres), dtype=np.float32).reshape(2, -1)
    contrato = construir_contrato_hce(
        resolucion=32,
        profundidad_max=5,
        estadisticas=estadisticas_features(matriz, nombres),
        n_modelos_train=2,
        conteo_train_esperado=9843,
        auditoria_train_completa=False,
    )

    assert contrato["autorizado_para_entrenamiento"] is False
    with pytest.raises(RuntimeError, match="no autoriza entrenamiento"):
        validar_contrato_hce(contrato, resolucion=32)


def _guardar_modelo_sintetico(
    data_root: Path, categoria: str, model_id: str, seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    puntos = rng.uniform(-0.95, 0.95, size=(500, 3)).astype(np.float32)
    normales = rng.normal(size=(500, 3)).astype(np.float32)
    normales /= np.clip(
        np.linalg.norm(normales, axis=1, keepdims=True), 1e-12, None,
    )
    raiz = construir_octree(puntos, normales, profundidad_max=5)
    destino = data_root / "octrees_32" / categoria / "train" / f"{model_id}.npz"
    destino.parent.mkdir(parents=True, exist_ok=True)
    guardar_octree_disperso(
        raiz,
        destino,
        etiqueta=CLASE_A_INDICE[categoria],
        profundidad_max=5,
        metadatos={
            "model_id": model_id,
            "categoria": categoria,
            "split": "train",
            "semilla_muestreo": seed,
            "n_puntos_muestreo": len(puntos),
            "archivo_origen_sha256": f"sha-{model_id}",
        },
    )


def test_auditoria_train_valida_metadatos_y_genera_contrato(tmp_path: Path):
    _guardar_modelo_sintetico(tmp_path, "airplane", "airplane_0001", 10)
    _guardar_modelo_sintetico(tmp_path, "chair", "chair_0001", 20)

    auditoria = auditar_resolucion(
        tmp_path,
        32,
        conteo_esperado=2,
        categorias_esperadas=["airplane", "chair"],
    )

    assert auditoria["auditoria_train_completa"] is True
    assert auditoria["n_archivos_validos"] == 2
    assert auditoria["n_errores"] == 0
    contrato = auditoria["contrato_features"]
    assert contrato["autorizado_para_entrenamiento"] is True
    validar_contrato_hce(contrato, resolucion=32)
