import json
from pathlib import Path

import pytest

from preprocesar_octrees import CLASES_MODELNET40
from validar_resultados_hce import (
    cargar_json_estricto,
    validar_archivos,
    validar_resultado_hce,
)


ROOT = Path(__file__).resolve().parents[1]


def _cargar_contrato() -> dict:
    ruta = ROOT / "resultados" / "objetivo2" / "contrato_hce_R32.json"
    with ruta.open("r", encoding="utf-8") as archivo:
        return json.load(archivo)


def _resultado_modelo(n_test: int, soportes: list[int]) -> dict:
    matriz = [[0 for _ in CLASES_MODELNET40] for _ in CLASES_MODELNET40]
    reporte = {}
    for indice, (clase, soporte) in enumerate(zip(CLASES_MODELNET40, soportes)):
        matriz[indice][indice] = soporte
        reporte[clase] = {
            "precision": 1.0,
            "recall": 1.0,
            "f1-score": 1.0,
            "support": float(soporte),
        }
    reporte["accuracy"] = 1.0
    for promedio in ("macro avg", "weighted avg"):
        reporte[promedio] = {
            "precision": 1.0,
            "recall": 1.0,
            "f1-score": 1.0,
            "support": float(n_test),
        }
    return {
        "val_acc": 1.0,
        "test_acc": 1.0,
        "tiempo_inferencia_total_s": 2.468,
        "tiempo_inferencia_promedio_ms": 1.0,
        "tamano_modelo_mb": 1.5,
        "reporte_clasificacion": reporte,
        "matriz_confusion": matriz,
    }


def _resultado_valido(contrato: dict) -> dict:
    n_test = 2468
    soportes = [62] * 28 + [61] * 12
    svm = {
        **_resultado_modelo(n_test, soportes),
        "mejores_hiperparametros": {"C": 10, "gamma": "scale"},
        "tiempo_busqueda_hiperparam_s": 10.0,
    }
    importancia = 1.0 / contrato["dimension_modelo"]
    rf = {
        **_resultado_modelo(n_test, soportes),
        "n_estimators": 100,
        "max_depth": None,
        "tiempo_entrenamiento_s": 5.0,
        "feature_importances": [
            {"feature": nombre, "importancia": importancia}
            for nombre in contrato["orden_modelo"]
        ],
    }
    return {
        "schema_name": "hce-training-results",
        "schema_version": "1.0.0",
        "resolucion": 32,
        "profundidad_octree": 5,
        "contrato_features": "resultados/objetivo2/contrato_hce_R32.json",
        "contract_version": contrato["contract_version"],
        "features_excluidas": contrato["caracteristicas_excluidas"],
        "orden_features_modelo": contrato["orden_modelo"],
        "n_features": contrato["dimension_modelo"],
        "n_train_total": 9843,
        "n_train": 8858,
        "n_val": 985,
        "n_test": n_test,
        "seed": 42,
        "svm": svm,
        "random_forest": rf,
    }


def test_resultado_hce_completo_es_valido():
    contrato = _cargar_contrato()

    errores = validar_resultado_hce(_resultado_valido(contrato), contrato, 32)

    assert errores == []


def test_resultado_hce_detecta_inconsistencias_cruzadas():
    contrato = _cargar_contrato()
    resultado = _resultado_valido(contrato)
    resultado["n_train_total"] = 9842
    resultado["svm"]["matriz_confusion"][0][0] -= 1
    resultado["random_forest"]["feature_importances"][0]["feature"] = "inventada"

    errores = validar_resultado_hce(resultado, contrato, 32)

    assert any("n_train_total" in error for error in errores)
    assert any("matriz_confusion suma 2467" in error for error in errores)
    assert any("no coincide con las características" in error for error in errores)


def test_resultado_hce_reporta_importancia_mal_formada_sin_fallar():
    contrato = _cargar_contrato()
    resultado = _resultado_valido(contrato)
    resultado["random_forest"]["feature_importances"][0] = {
        "feature": None,
        "importancia": "mucho",
    }

    errores = validar_resultado_hce(resultado, contrato, 32)

    assert any("contiene entradas inválidas" in error for error in errores)


def test_carga_json_estricta_rechaza_nan(tmp_path: Path):
    ruta = tmp_path / "resultado.json"
    ruta.write_text('{"test_acc": NaN}', encoding="utf-8")

    with pytest.raises(ValueError, match="Constante JSON no válida"):
        cargar_json_estricto(ruta)


def test_informe_consolidado_exige_resultado_y_contrato(tmp_path: Path):
    contrato = _cargar_contrato()
    (tmp_path / "contrato_hce_R32.json").write_text(
        json.dumps(contrato), encoding="utf-8",
    )
    (tmp_path / "resumen_hce_R32.json").write_text(
        json.dumps(_resultado_valido(contrato)), encoding="utf-8",
    )

    informe_valido = validar_archivos(tmp_path, [32])
    informe_incompleto = validar_archivos(tmp_path, [32, 64])

    assert informe_valido["validacion_completa"] is True
    assert informe_incompleto["validacion_completa"] is False
    assert informe_incompleto["validaciones"][1]["errores"]
