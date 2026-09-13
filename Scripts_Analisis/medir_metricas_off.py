"""
medir_metricas_off.py
======================
Procesa TODOS los archivos .off de ModelNet40 y calcula las metricas
de estructura del octree REAL, tiempo de procesamiento y almacenamiento
real en disco, para la tabla del capitulo de metodologia.

CORRECCION respecto a versiones anteriores: ya no se voxeliza a una
rejilla densa de un solo canal para "estimar" memoria. Se construye
el octree REAL (con poda, ver octree_real.py) para cada objeto, se
guarda en disco con su estructura jerarquica completa, y se miden
directamente:
  - Numero de nodos totales del arbol (internos + hojas)
  - Numero de hojas ocupadas
  - Tamaño REAL del archivo .npz en disco (no estimado)
  - Porcentaje de ocupacion en el nivel hoja

Metricas calculadas por objeto:
  - Numero de vertices y caras de la malla original
  - Tiempo de lectura + normalizacion + muestreo (ms)
  - Tiempo de construccion del arbol real, 32^3 y 64^3 (ms)
  - Nodos totales y hojas ocupadas del arbol, 32^3 y 64^3
  - Porcentaje de ocupacion en la hoja, 32^3 y 64^3
  - Tamaño REAL del archivo .npz guardado (KB), 32^3 y 64^3

Salida:
  - resultados/metricas_off_completo.csv   (todas las muestras)
  - resultados/metricas_off_resumen.csv    (estadisticas por clase)
  - resultados/metricas_off_global.json    (estadisticas globales)

Uso:
    python medir_metricas_off.py
    python medir_metricas_off.py --n_muestras 100   # muestra rapida
    python medir_metricas_off.py --split train
"""

import sys
import csv
import json
import time
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

RAIZ_PROYECTO = Path(__file__).parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO / "fase2_octree"))

from octree import leer_off, normalizar_malla, muestrear_superficie_con_normales
from octree_real import (
    construir_octree, guardar_octree_disperso, recolectar_hojas,
    contar_nodos_totales,
)

RAIZ_DATASET   = RAIZ_PROYECTO / "Dataset" / "ModelNet40"
DIR_RESULTADOS = RAIZ_PROYECTO / "resultados"
DIR_TEMP_NPZ   = RAIZ_PROYECTO / "data" / "_temp_medicion"  # npz temporales de medicion
N_PUNTOS_MUESTREO = 20000
N_PROCESOS = 10
SEED = 42
PROFUNDIDAD_POR_RESOLUCION = {32: 5, 64: 6}
RESOLUCIONES = [32, 64]

CLASES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf",
    "bottle", "bowl", "car", "chair", "cone",
    "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
    "laptop", "mantel", "monitor", "night_stand", "person",
    "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent",
    "toilet", "tv_stand", "vase", "wardrobe", "xbox",
]


# ──────────────────────────────────────────────────────────────
# PROCESAMIENTO DE UN ARCHIVO
# ──────────────────────────────────────────────────────────────

def procesar_archivo(args: tuple) -> dict:
    ruta, clase, split = args
    nombre = Path(ruta).stem

    try:
        # Lectura + normalizacion + muestreo (una sola vez, compartido
        # entre resoluciones)
        t0 = time.perf_counter()
        verts_raw, caras = leer_off(ruta)
        verts = normalizar_malla(verts_raw)
        t_preproceso_ms = (time.perf_counter() - t0) * 1000

        n_v = len(verts_raw)
        n_c = len(caras) if caras is not None else 0

        rng = np.random.default_rng(SEED)
        t0 = time.perf_counter()
        pts, normales = muestrear_superficie_con_normales(
            verts, caras, N_PUNTOS_MUESTREO, rng,
        )
        t_muestreo_ms = (time.perf_counter() - t0) * 1000

        fila = {
            "nombre": nombre, "clase": clase, "split": split,
            "n_vertices": n_v, "n_caras": n_c,
            "t_preproceso_ms": round(t_preproceso_ms, 3),
            "t_muestreo_ms": round(t_muestreo_ms, 3),
            "error": None,
        }

        for R in RESOLUCIONES:
            profundidad_max = PROFUNDIDAD_POR_RESOLUCION[R]

            t0 = time.perf_counter()
            raiz = construir_octree(pts, normales, profundidad_max=profundidad_max)
            t_arbol_ms = (time.perf_counter() - t0) * 1000

            n_nodos = contar_nodos_totales(raiz)
            hojas = recolectar_hojas(raiz)
            n_hojas = len(hojas)
            pct_ocup_hoja = 100.0 * n_hojas / (R ** 3)

            # Guardar en disco temporal para medir el TAMAÑO REAL del
            # archivo con estructura jerarquica completa (no estimado)
            dir_split = DIR_TEMP_NPZ / f"R{R}" / clase / split
            dir_split.mkdir(parents=True, exist_ok=True)
            ruta_npz = dir_split / f"{nombre}.npz"

            t0 = time.perf_counter()
            guardar_octree_disperso(raiz, str(ruta_npz), etiqueta=0,
                                    profundidad_max=profundidad_max)
            t_guardado_ms = (time.perf_counter() - t0) * 1000

            tam_archivo_kb = ruta_npz.stat().st_size / 1024

            fila[f"t_arbol_{R}_ms"]      = round(t_arbol_ms, 3)
            fila[f"t_guardado_{R}_ms"]   = round(t_guardado_ms, 3)
            fila[f"nodos_totales_{R}"]   = n_nodos
            fila[f"hojas_ocupadas_{R}"]  = n_hojas
            fila[f"ocup_hoja_pct_{R}"]   = round(pct_ocup_hoja, 4)
            fila[f"tam_npz_kb_{R}"]      = round(tam_archivo_kb, 3)

            # Borrar el npz temporal (solo se necesitaba para medir tamaño)
            ruta_npz.unlink()

        fila["t_total_ms"] = round(
            t_preproceso_ms + t_muestreo_ms
            + sum(fila[f"t_arbol_{R}_ms"] for R in RESOLUCIONES)
            + sum(fila[f"t_guardado_{R}_ms"] for R in RESOLUCIONES), 3,
        )

        return fila

    except Exception as e:
        return {"nombre": nombre, "clase": clase, "split": split, "error": str(e)}


# ──────────────────────────────────────────────────────────────
# RECOLECCION Y PROCESAMIENTO MASIVO
# ──────────────────────────────────────────────────────────────

def recolectar(split: str, n_muestras: int = None) -> list:
    tareas = []
    for clase in CLASES:
        carpeta = RAIZ_DATASET / clase / split
        if not carpeta.exists():
            continue
        archivos = sorted(carpeta.glob("*.off"))
        if n_muestras:
            archivos = archivos[:max(1, n_muestras // len(CLASES))]
        for f in archivos:
            tareas.append((str(f), clase, split))
    return tareas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test", "ambos"])
    parser.add_argument("--n_muestras", type=int, default=None)
    args = parser.parse_args()

    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    DIR_TEMP_NPZ.mkdir(parents=True, exist_ok=True)

    splits = ["train", "test"] if args.split == "ambos" else [args.split]

    print("=" * 70)
    print("  METRICAS DE PROCESAMIENTO .off -> Octree REAL")
    print("  (nodos, hojas, tamaño real de archivo, tiempos)")
    print("=" * 70)

    todos_resultados = []

    for split in splits:
        tareas = recolectar(split, args.n_muestras)
        print(f"\n[{split.upper()}] {len(tareas)} archivos a procesar...")

        t_inicio = time.time()
        with ProcessPoolExecutor(max_workers=N_PROCESOS) as executor:
            futuros = {executor.submit(procesar_archivo, t): t for t in tareas}
            barra = tqdm(as_completed(futuros), total=len(futuros), ncols=80)
            for futuro in barra:
                todos_resultados.append(futuro.result())
        t_total = time.time() - t_inicio

        n_ok  = sum(1 for r in todos_resultados if r.get("error") is None and r["split"] == split)
        n_err = sum(1 for r in todos_resultados if r.get("error") is not None and r["split"] == split)
        print(f"  Completado en {t_total:.1f}s | OK: {n_ok} | Errores: {n_err}")

    # ── Guardar CSV completo ──
    validos = [r for r in todos_resultados if r.get("error") is None]
    campos = ["nombre", "clase", "split", "n_vertices", "n_caras",
              "t_preproceso_ms", "t_muestreo_ms", "t_total_ms"]
    for R in RESOLUCIONES:
        campos += [f"t_arbol_{R}_ms", f"t_guardado_{R}_ms",
                  f"nodos_totales_{R}", f"hojas_ocupadas_{R}",
                  f"ocup_hoja_pct_{R}", f"tam_npz_kb_{R}"]

    csv_completo = DIR_RESULTADOS / "metricas_off_completo.csv"
    with open(csv_completo, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for r in validos:
            w.writerow({k: r.get(k, "") for k in campos})
    print(f"\n[CSV] Guardado: {csv_completo.name} ({len(validos)} filas)")

    # ── Resumen por clase ──
    from collections import defaultdict
    por_clase = defaultdict(list)
    for r in validos:
        por_clase[r["clase"]].append(r)

    csv_resumen = DIR_RESULTADOS / "metricas_off_resumen.csv"
    campos_resumen = ["clase", "n", "t_total_media_ms"]
    for R in RESOLUCIONES:
        campos_resumen += [f"nodos_media_{R}", f"hojas_media_{R}", f"ocup_pct_media_{R}",
                           f"tam_npz_kb_media_{R}"]

    with open(csv_resumen, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos_resumen)
        w.writeheader()
        for clase, muestras in por_clase.items():
            fila = {
                "clase": clase, "n": len(muestras),
                "t_total_media_ms": round(np.mean([m["t_total_ms"] for m in muestras]), 2),
            }
            for R in RESOLUCIONES:
                fila[f"nodos_media_{R}"] = round(np.mean([m[f"nodos_totales_{R}"] for m in muestras]), 1)
                fila[f"hojas_media_{R}"] = round(np.mean([m[f"hojas_ocupadas_{R}"] for m in muestras]), 1)
                fila[f"ocup_pct_media_{R}"] = round(np.mean([m[f"ocup_hoja_pct_{R}"] for m in muestras]), 4)
                fila[f"tam_npz_kb_media_{R}"] = round(np.mean([m[f"tam_npz_kb_{R}"] for m in muestras]), 3)
            w.writerow(fila)
    print(f"[CSV] Guardado: {csv_resumen.name}")

    # ── Resumen global ──
    resumen_global = {"n_total": len(validos), "n_errores": len(todos_resultados) - len(validos)}
    resumen_global["t_total_ms"] = {
        "media": round(float(np.mean([r["t_total_ms"] for r in validos])), 4),
        "std":   round(float(np.std([r["t_total_ms"] for r in validos])), 4),
        "min":   round(float(np.min([r["t_total_ms"] for r in validos])), 4),
        "max":   round(float(np.max([r["t_total_ms"] for r in validos])), 4),
    }
    for R in RESOLUCIONES:
        nodos = [r[f"nodos_totales_{R}"] for r in validos]
        hojas = [r[f"hojas_ocupadas_{R}"] for r in validos]
        ocup  = [r[f"ocup_hoja_pct_{R}"] for r in validos]
        tam   = [r[f"tam_npz_kb_{R}"] for r in validos]

        resumen_global[f"R{R}"] = {
            "nodos_media": round(float(np.mean(nodos)), 1),
            "nodos_std":   round(float(np.std(nodos)), 1),
            "hojas_media": round(float(np.mean(hojas)), 1),
            "hojas_std":   round(float(np.std(hojas)), 1),
            "ocup_pct_media": round(float(np.mean(ocup)), 4),
            "ocup_pct_std":   round(float(np.std(ocup)), 4),
            "tam_npz_kb_media": round(float(np.mean(tam)), 3),
            "tam_npz_kb_total_dataset_mb": round(float(np.sum(tam)) / 1024, 2),
            "mem_densa_equivalente_total_mb": round(
                len(validos) * 4 * (R**3) * 4 / 1024 / 1024, 1
            ),
        }
        factor = resumen_global[f"R{R}"]["mem_densa_equivalente_total_mb"] * 1024 / \
                 max(resumen_global[f"R{R}"]["tam_npz_kb_total_dataset_mb"] * 1024, 0.001)
        resumen_global[f"R{R}"]["factor_ahorro_real"] = round(factor, 1)

    ruta_json = DIR_RESULTADOS / "metricas_off_global.json"
    with open(ruta_json, "w") as f:
        json.dump(resumen_global, f, indent=2)
    print(f"[JSON] Guardado: {ruta_json.name}")

    # ── Resumen en consola ──
    print("\n" + "=" * 70)
    print("  RESUMEN GLOBAL (octree real, medido)")
    print("=" * 70)
    print(f"  Objetos procesados : {resumen_global['n_total']:,}")
    print(f"  Errores            : {resumen_global['n_errores']}")
    print(f"  Tiempo total/objeto: media={resumen_global['t_total_ms']['media']:.2f} ms, "
          f"max={resumen_global['t_total_ms']['max']:.2f} ms")

    for R in RESOLUCIONES:
        g = resumen_global[f"R{R}"]
        print(f"\n  --- Resolucion {R}^3 ---")
        print(f"    Nodos totales (media)   : {g['nodos_media']:.0f} (±{g['nodos_std']:.0f})")
        print(f"    Hojas ocupadas (media)  : {g['hojas_media']:.0f} (±{g['hojas_std']:.0f})")
        print(f"    Ocupacion hoja (media)  : {g['ocup_pct_media']:.3f}% (±{g['ocup_pct_std']:.3f}%)")
        print(f"    Tamaño .npz (media)     : {g['tam_npz_kb_media']:.3f} KB")
        print(f"    Tamaño dataset (real)   : {g['tam_npz_kb_total_dataset_mb']:.2f} MB")
        print(f"    Densa equivalente       : {g['mem_densa_equivalente_total_mb']:.1f} MB")
        print(f"    >>> Factor de ahorro REAL: {g['factor_ahorro_real']:.1f}x <<<")

    print("=" * 70)

    # Limpiar carpeta temporal
    import shutil
    shutil.rmtree(DIR_TEMP_NPZ, ignore_errors=True)


if __name__ == "__main__":
    main()
