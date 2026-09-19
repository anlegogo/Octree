import json
from pathlib import Path

from octree_real import OCTREE_FORMAT_VERSION, leer_metadatos_octree
from preprocesar_octrees import procesar_modelo, seleccionar_muestra_controlada


def test_muestra_controlada_es_balanceada_y_determinista():
    modelos = [
        {"model_id": f"{categoria}_{split}_{indice}",
         "categoria": categoria, "split": split}
        for split in ("train", "test")
        for categoria in ("airplane", "chair")
        for indice in range(3)
    ]
    muestra = seleccionar_muestra_controlada(modelos, 1)

    assert [modelo["model_id"] for modelo in muestra] == [
        "airplane_train_0", "chair_train_0",
        "airplane_test_0", "chair_test_0",
    ]


def test_pipeline_genera_npz_manifiesto_y_metricas(tmp_path: Path):
    fixture = Path(__file__).parent / "fixtures" / "tetrahedron.off"
    output_root = tmp_path / "data"
    tarea = {
        "ruta": str(fixture),
        "ruta_relativa": "chair/train/chair_control.off",
        "model_id": "chair_control",
        "categoria": "chair",
        "etiqueta": 8,
        "split": "train",
        "output_root": str(output_root),
        "manifest_root": str(output_root / "manifests"),
        "resoluciones": [32, 64],
        "n_puntos": 1000,
        "semilla_base": 42,
        "sobrescribir": False,
    }

    resultado = procesar_modelo(tarea)
    assert resultado["ok"] is True
    with Path(resultado["manifiesto"]).open(encoding="utf-8") as archivo:
        manifiesto = json.load(archivo)
    assert manifiesto["octree_format_version"] == OCTREE_FORMAT_VERSION
    assert set(manifiesto["salidas"]) == {"32", "64"}

    for resolucion, datos in manifiesto["salidas"].items():
        assert datos["equivalencia_antes_guardado"]["equivalente"] is True
        assert datos["equivalencia_despues_carga"]["equivalente"] is True
        assert len(datos["nodos_por_nivel"]) == (6 if resolucion == "32" else 7)
        assert datos["bytes_binarios_por_nodo"] == 18
        assert (
            datos["carga_binaria_sin_comprimir_bytes"]
            > datos["carga_binaria_nodos_sin_comprimir_bytes"]
        )
        ruta_npz = output_root / datos["archivo_npz"]
        assert ruta_npz.is_file()
        metadatos = leer_metadatos_octree(ruta_npz)
        assert metadatos["model_id"] == "chair_control"
        assert metadatos["resolucion"] == int(resolucion)
