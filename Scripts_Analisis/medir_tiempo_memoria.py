"""Mide un modelo con el mismo pipeline definitivo del objetivo 1.

El reporte separa tiempo de construccion, pico transitorio, memoria de
la estructura Python, payload binario sin comprimir, NPZ real y
referencia densa. No ejecuta HCE ni ningun clasificador.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO / "fase2_octree"))

from preprocesar_octrees import procesar_modelo  # noqa: E402
from validacion_objetivo1 import guardar_json_atomico  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--categoria", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--etiqueta", type=int, default=0)
    parser.add_argument("--n-puntos", type=int, default=20000)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument(
        "--salida", type=Path, default=None,
        help="JSON de salida; por defecto metricas_<modelo>.json",
    )
    args = parser.parse_args()
    ruta = args.off.resolve()
    if not ruta.is_file():
        raise FileNotFoundError(ruta)

    categoria = args.categoria or ruta.parent.parent.name or "desconocida"
    split = args.split or ruta.parent.name or "control"
    salida = args.salida or Path(f"metricas_{ruta.stem}.json")

    with tempfile.TemporaryDirectory(prefix="metricas_objetivo1_") as temporal:
        raiz_temporal = Path(temporal)
        tarea = {
            "ruta": str(ruta),
            "ruta_relativa": f"{categoria}/{split}/{ruta.name}",
            "model_id": ruta.stem,
            "categoria": categoria,
            "etiqueta": args.etiqueta,
            "split": split,
            "output_root": str(raiz_temporal / "data"),
            "manifest_root": str(raiz_temporal / "data" / "manifests"),
            "resoluciones": [32, 64],
            "n_puntos": args.n_puntos,
            "semilla_base": args.semilla,
            "sobrescribir": True,
        }
        resultado = procesar_modelo(tarea)
        with Path(resultado["manifiesto"]).open(encoding="utf-8") as archivo:
            reporte = json.load(archivo)

    # Los NPZ usados para medir fueron temporales; el JSON conserva sus
    # tamaños y hashes, pero no anuncia rutas inexistentes como artefactos.
    for datos in reporte["salidas"].values():
        datos.pop("archivo_npz", None)
        datos.pop("archivo_npz_sha256", None)
    reporte["tipo_reporte"] = "medicion_controlada_sin_artefactos_persistentes"
    guardar_json_atomico(reporte, salida)

    print(f"Modelo: {ruta.name}")
    for resolucion, datos in reporte["salidas"].items():
        print(
            f"R={resolucion}: nodos={datos['n_nodos_totales']:,}, "
            f"hojas={datos['n_hojas_ocupadas']:,}, "
            f"tiempo={datos['tiempo_construccion_ms']:.3f} ms, "
            f"equivalencia=OK"
        )
    print(f"Reporte: {salida.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
