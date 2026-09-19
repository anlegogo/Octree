"""Verifica y consolida las metricas finales del objetivo especifico 1.

Los manifiestos son la fuente de verdad. Este programa comprueba su
formato, la presencia e integridad de cada NPZ y la equivalencia en
32^3 y 64^3 antes de producir el resumen global de ModelNet40.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO / "fase2_octree"))

from octree_real import OCTREE_FORMAT_VERSION, leer_metadatos_octree  # noqa: E402
from validacion_objetivo1 import (  # noqa: E402
    MANIFEST_FORMAT_VERSION,
    guardar_json_atomico,
    sha256_archivo,
)

CONTEOS_OFICIALES = {"train": 9843, "test": 2468}
METRICAS_NUMERICAS = [
    "n_nodos_totales",
    "n_hojas_ocupadas",
    "porcentaje_ocupacion_hoja",
    "tiempo_construccion_ms",
    "memoria_pico_construccion_bytes",
    "memoria_estructura_python_bytes",
    "carga_binaria_sin_comprimir_bytes",
    "tamano_npz_bytes",
]


def _estadisticas(valores: list[float]) -> dict:
    array = np.asarray(valores, dtype=np.float64)
    return {
        "n": int(array.size),
        "media": float(array.mean()),
        "desviacion_estandar": float(array.std(ddof=0)),
        "minimo": float(array.min()),
        "mediana": float(np.median(array)),
        "maximo": float(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifests-dir", type=Path,
        default=RAIZ_PROYECTO / "data" / "manifests",
    )
    parser.add_argument(
        "--output-root", type=Path, default=RAIZ_PROYECTO / "data",
        help="Raiz contra la cual se resuelven archivo_npz de los manifiestos",
    )
    parser.add_argument(
        "--resultados-dir", type=Path,
        default=RAIZ_PROYECTO / "resultados" / "objetivo1",
    )
    parser.add_argument(
        "--permitir-incompleto", action="store_true",
        help="Permite consolidar una corrida corta de diagnostico",
    )
    args = parser.parse_args()

    rutas = sorted(args.manifests_dir.rglob("*.json"))
    if not rutas:
        raise FileNotFoundError(f"No hay manifiestos en {args.manifests_dir}")

    filas = []
    errores = []
    modelos = set()
    conteos_split = Counter()
    acumulados = defaultdict(lambda: defaultdict(list))

    for ruta_manifiesto in rutas:
        try:
            with ruta_manifiesto.open(encoding="utf-8") as archivo:
                manifiesto = json.load(archivo)
            if manifiesto["manifest_format_version"] != MANIFEST_FORMAT_VERSION:
                raise ValueError("version de manifiesto incompatible")
            if manifiesto["octree_format_version"] != OCTREE_FORMAT_VERSION:
                raise ValueError("version de octree incompatible")

            clave_modelo = (
                manifiesto["categoria"], manifiesto["split"], manifiesto["model_id"],
            )
            if clave_modelo in modelos:
                raise ValueError("manifiesto duplicado")
            modelos.add(clave_modelo)
            conteos_split[manifiesto["split"]] += 1

            if set(manifiesto["salidas"]) != {"32", "64"}:
                raise ValueError("el manifiesto no contiene exactamente R32 y R64")

            for resolucion_texto, datos in manifiesto["salidas"].items():
                for campo_equivalencia in (
                    "equivalencia_antes_guardado", "equivalencia_despues_carga",
                ):
                    if not datos[campo_equivalencia]["equivalente"]:
                        raise ValueError(f"fallo de {campo_equivalencia}")

                ruta_npz = args.output_root / datos["archivo_npz"]
                if not ruta_npz.is_file():
                    raise FileNotFoundError(f"falta {ruta_npz}")
                if sha256_archivo(ruta_npz) != datos["archivo_npz_sha256"]:
                    raise ValueError(f"SHA-256 no coincide para {ruta_npz}")
                metadatos_npz = leer_metadatos_octree(ruta_npz)
                if metadatos_npz["model_id"] != manifiesto["model_id"]:
                    raise ValueError("model_id del NPZ no coincide con el manifiesto")
                if metadatos_npz["resolucion"] != int(resolucion_texto):
                    raise ValueError("resolucion del NPZ no coincide con el manifiesto")

                fila = {
                    "model_id": manifiesto["model_id"],
                    "categoria": manifiesto["categoria"],
                    "split": manifiesto["split"],
                    "resolucion": int(resolucion_texto),
                    "nodos_por_nivel": "|".join(map(str, datos["nodos_por_nivel"])),
                    **{metrica: datos[metrica] for metrica in METRICAS_NUMERICAS},
                    "densa_carga_sin_comprimir_bytes": datos["referencia_densa"][
                        "carga_sin_comprimir_bytes"
                    ],
                    "densa_tamano_npz_comprimido_bytes": datos["referencia_densa"][
                        "tamano_npz_comprimido_bytes"
                    ],
                    "equivalencia": True,
                }
                filas.append(fila)
                grupo = (manifiesto["split"], int(resolucion_texto))
                for metrica in METRICAS_NUMERICAS:
                    acumulados[grupo][metrica].append(float(datos[metrica]))

        except Exception as exc:
            errores.append({
                "manifiesto": str(ruta_manifiesto),
                "error": f"{type(exc).__name__}: {exc}",
            })

    completo = (
        dict(conteos_split) == CONTEOS_OFICIALES
        and len(modelos) == sum(CONTEOS_OFICIALES.values())
        and len(filas) == 2 * sum(CONTEOS_OFICIALES.values())
        and not errores
    )
    fallo_por_incompleto = not completo and not args.permitir_incompleto

    args.resultados_dir.mkdir(parents=True, exist_ok=True)
    ruta_csv = args.resultados_dir / "metricas_modelnet40_verificadas.csv"
    with ruta_csv.open("w", newline="", encoding="utf-8") as archivo:
        campos = list(filas[0]) if filas else []
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)

    resumen_grupos = {}
    for (split, resolucion), metricas in sorted(acumulados.items()):
        resumen_grupos[f"{split}_R{resolucion}"] = {
            metrica: _estadisticas(valores)
            for metrica, valores in metricas.items()
        }
    resumen = {
        "manifest_format_version": MANIFEST_FORMAT_VERSION,
        "octree_format_version": OCTREE_FORMAT_VERSION,
        "modelnet40_completo": completo,
        "conteos_por_split": dict(conteos_split),
        "n_modelos_unicos": len(modelos),
        "n_registros_modelo_resolucion": len(filas),
        "n_errores": len(errores),
        "errores": errores,
        "estadisticas": resumen_grupos,
    }
    ruta_resumen = args.resultados_dir / "resumen_metricas_modelnet40.json"
    guardar_json_atomico(resumen, ruta_resumen)
    print(f"ModelNet40 completo: {'SI' if completo else 'NO (diagnostico)'}")
    print(f"CSV: {ruta_csv}")
    print(f"Resumen: {ruta_resumen}")
    if fallo_por_incompleto:
        print(
            "ERROR: la corrida no contiene ModelNet40 completo. "
            "Los resultados se conservaron para diagnostico."
        )
    if errores:
        return 1
    return 0 if completo or args.permitir_incompleto else 1


if __name__ == "__main__":
    raise SystemExit(main())
